"""Local dashboard contract checks; synthetic machine copies, no inference."""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import dashboard as d


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent.parent)
        self.root=Path(self.temp.name)
        self.old_output=d.OUTPUT;d.OUTPUT=self.root/'exports'
        self.client=d.app.test_client()
        self.bank=d.load_bank().iloc[:20]
        self.sources=[]
        for m in range(3):
            machine=self.root/f'pc{m}'
            self.sources.append({'name':f'PC {m+1}','path':str(machine)})
            for mode in ['original','questions']:
                folder=machine/f'arbitrary_{mode}';folder.mkdir(parents=True)
                rows=[]
                from question_order import arrange, save_order
                frame=arrange(self.bank,mode,42)
                for pos,(_,q) in enumerate(frame.iterrows(),1):
                    rows.append(dict(question_id=q['Question ID'],question=q['Question'],answer='fixture',
                                     model_name='fixture:3b',status='ok',error='',execution_position=pos,
                                     ordering_mode=mode,latency_s=2+m,emissions_kg=.001*(m+1),
                                     energy_kwh=.002,cpu_energy_kwh=.001,ram_energy_kwh=.001,gpu_energy_kwh=0))
                with (folder/'answers.csv').open('w',newline='',encoding='utf-8') as f:
                    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
                with (folder/'summary.csv').open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=['num_queries']);w.writeheader();w.writerow({'num_queries':20})
                (folder/'run_status.json').write_text(json.dumps({'status':'completed'}))
                save_order(frame,folder)
        self.snap={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob('*') if p.is_file()}
        response=self.client.post('/api/load',json={'machines':self.sources})
        self.assertEqual(response.status_code,200,response.data)
        loaded=response.get_json();self.dataset=loaded['dataset']
        self.ids=[r['id'] for machine in loaded['machines'] for r in machine['runs']]
        self.payload={'dataset':self.dataset,'ids':self.ids}

    def tearDown(self):
        for path,digest in self.snap.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),digest)
        d.OUTPUT=self.old_output;d.CATALOGUES.clear();d.ARTIFACTS.clear()
        self.temp.cleanup()

    def test_tables_matching_and_averages(self):
        response=self.client.post('/api/analyse',json=self.payload)
        self.assertEqual(response.status_code,200,response.data)
        result=response.get_json()
        self.assertEqual(len(result['questions']),120)
        self.assertEqual(len(result['followups']),5)
        self.assertTrue(all(r['mean_emissions_g']==2 for r in result['averages']))
        for row in result['followups']:
            gaps=[v for k,v in row.items() if k.endswith(' gap')]
            self.assertTrue(all(v is None for v in gaps) if row['turn']==1 else all(v==0 for v in gaps))

    def test_custom_graphs_and_all_exports(self):
        from pypdf import PdfReader
        for kind in ['execution','range','average','report_individual','report_comparison']:
            response=self.client.post('/api/graph',json={**self.payload,'kind':kind,'page':1})
            self.assertEqual(response.status_code,200,response.data)
            panels=response.get_json()['panels'];self.assertEqual(len(panels),3)
            for panel in panels:
                with self.client.get(panel['url']) as image_response:
                    self.assertEqual(image_response.status_code,200)
        for kind in ['csv','xlsx','comparison','custom','individual','side-by-side']:
            response=self.client.post('/api/export',json={**self.payload,'kind':kind,'run':self.ids[0],
                                                        'graph_kind':'range','page':1})
            self.assertEqual(response.status_code,200,response.data)
            result=response.get_json();token=result['url'].split('/')[-1];path=d.ARTIFACTS[token]
            self.assertGreater(path.stat().st_size,0)
            if path.suffix=='.pdf':self.assertGreater(len(PdfReader(path).pages),0)

    def test_reject_mixed_modes_changed_text_and_unknown_ids(self):
        item=d.CATALOGUES[self.dataset][self.ids[-1]]
        original=item['run']['metadata']['Pipeline']
        item['run']['metadata']['Pipeline']='RAG'
        self.assertEqual(self.client.post('/api/analyse',json=self.payload).status_code,400)
        item['run']['metadata']['Pipeline']=original
        q=next(iter(item['run']['questions']));record=item['run']['questions'].pop(q)
        item['run']['questions'][q+' changed']=record
        self.assertEqual(self.client.post('/api/analyse',json=self.payload).status_code,400)
        self.assertEqual(self.client.post('/api/analyse',json={**self.payload,'ids':['unknown']}).status_code,400)

    def test_loopback_and_origin_restrictions(self):
        self.assertEqual(self.client.get('/',headers={'Host':'example.com'}).status_code,403)
        self.assertEqual(self.client.post('/api/load',json={},headers={'Origin':'https://example.com'}).status_code,403)
        self.assertEqual(self.client.get('/artifact/not-real').status_code,404)

    def test_rag_phases_and_missing_machine_panel(self):
        # Convert only these synthetic fixtures; no real experiment data is changed.
        for path in self.root.glob('pc*/*/answers.csv'):
            with path.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
            for row in rows:
                row['total_emissions_kg']=row.pop('emissions_kg')
                row['total_latency_s']=row.pop('latency_s')
                row.update(retrieval_emissions_kg=.0002,generation_emissions_kg=float(row['total_emissions_kg'])-.0002,
                           retrieval_latency_s=.1,generation_latency_s=float(row['total_latency_s'])-.1,
                           retrieval_energy_kwh=.0005,generation_energy_kwh=.0015)
            with path.open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        self.snap={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.snap}
        response=self.client.post('/api/load',json={'machines':self.sources})
        loaded=response.get_json()
        payload={'dataset':loaded['dataset'],'ids':[r['id'] for m in loaded['machines'] for r in m['runs']]}
        response=self.client.post('/api/graph',json={**payload,'kind':'phases','page':1})
        self.assertEqual(response.status_code,200,response.data)
        self.assertTrue(all('url' in p for p in response.get_json()['panels']))
        response=self.client.post('/api/graph',json={**payload,'ids':payload['ids'][:1],'kind':'phases','page':1})
        self.assertEqual(response.status_code,200,response.data)
        self.assertEqual(sum('missing' in p for p in response.get_json()['panels']),2)

    def test_source_changes_require_reload(self):
        path=self.root/'pc0'/'arbitrary_original'/'answers.csv'
        before=path.read_bytes()
        try:
            path.write_bytes(before+b'\n')
            response=self.client.post('/api/analyse',json=self.payload)
            self.assertEqual(response.status_code,400)
            self.assertIn('Reload',response.get_json()['error'])
        finally:path.write_bytes(before)


if __name__=='__main__':unittest.main()
