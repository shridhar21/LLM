"""Hybrid RAG tests using real FAISS/LangChain, but no model downloads or live inference."""
import ast
import contextlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import faiss
import numpy as np
import pandas as pd

import advanced_rag as rag
import cancellation
from test_cancellation import load_functions, Tracker

ROOT = Path(__file__).resolve().parent


def assets_fixture():
    children, parents = [], []
    for i, (topic, text) in enumerate([
        ('Energy', 'Solar panels generate electricity using sunlight. Solar energy is renewable.'),
        ('Wind', 'Wind turbines produce electricity from moving air.'),
        ('Water', 'Water storage and rivers support agriculture and irrigation.'),
        ('Food', 'Food preparation includes baking bread and cooking vegetables.'),
    ]):
        c, p = rag.document_records('# ' + topic + '\n' + text, f'{i}.txt', str(i))
        children.extend(c)
        parents.extend(p)
    vectors = np.array([[1, 0, 0], [.9, .1, 0], [0, 1, 0], [0, 0, 1]], dtype='float32')
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(3)
    index.add(vectors)
    ids = [c['chunk_id'] for c in children]
    parent_map = rag.validate_assets(index, children, parents, vectors, ids)
    return rag.RagAssets(index, children, parent_map, vectors, ids, rag.langchain_bm25(children))


class AdvancedRagTests(unittest.TestCase):
    def test_structural_headings_and_code_preservation(self):
        sections = rag.structural_sections('Preamble.\n# Introduction\nFirst text.\nII. Methods\nSecond text.\n'
                                           'RESULTS\nThird text.\n```\n# not a heading\n```')
        self.assertEqual([s[0] for s in sections], ['Document', 'Introduction', 'II. Methods', 'RESULTS'])
        self.assertIn('# not a heading', sections[-1][1])
        self.assertEqual(rag.normalize_heading_text('R E S U L T S'), 'RESULTS')

    def test_chunk_sizes_parent_links_and_sentence_overlap(self):
        text = '# Chapter\n' + '\n\n'.join(' '.join([f'Sentence{i}'] + ['word'] * 68) + '.' for i in range(35))
        children, parents = rag.document_records(text, 'source.txt', 'doc')
        self.assertGreater(len(parents), 1)
        self.assertTrue(all(len(rag.tokens(c['text'])) <= 180 for c in children))
        self.assertTrue(all(len(rag.tokens(p['text'])) <= 900 for p in parents))
        parent_ids = {p['parent_id'] for p in parents}
        self.assertTrue(all(c['parent_id'] in parent_ids and c['section'] == 'Chapter' for c in children))
        self.assertEqual(len(set(c['chunk_id'] for c in children)), len(children))
        same_parent = [c for c in children if c['parent_id'] == parents[0]['parent_id']]
        self.assertIn('Sentence1', same_parent[0]['text'])
        self.assertIn('Sentence1', same_parent[1]['text'])
        c, _ = rag.document_records(' '.join(['long'] * 1500), 'long.txt', 'long')
        self.assertTrue(all(len(rag.tokens(record['text'])) <= 180 for record in c))

    def test_rrf_and_mmr(self):
        fused = rag.reciprocal_rank_fusion([('A', .8), ('B', .7)], [('B', 2), ('C', 1)])
        self.assertEqual(fused[0]['chunk_id'], 'B')
        self.assertAlmostEqual(fused[0]['rrf_score'], 1 / 62 + 1 / 61)
        self.assertEqual(len(fused), 3)
        vectors = np.array([[1, 0], [1, 0], [.8, .6]], dtype='float32')
        # With this fixture the next-most-relevant identical vector still wins:
        # check exact scoring against the agreed .75/.25 rule, not an assumed diversity outcome.
        selected = rag.mmr_select([{'chunk_id': name} for name in 'ABC'], np.array([1, 0]), vectors, list('ABC'))
        self.assertEqual([c['chunk_id'] for c in selected], ['A', 'B', 'C'])
        vectors = np.array([[1, 0], [.9, .4358899], [.88, -.4749737]], dtype='float32')
        selected = rag.mmr_select([{'chunk_id': name} for name in 'ABC'], vectors[1], vectors, list('ABC'))
        self.assertEqual(selected[0]['chunk_id'], 'B')

    def test_evidence_gate_rules_and_threshold(self):
        self.assertEqual(rag.evidence_gate([], [('A', 1)], []), (False, 'no_ranked_context'))
        self.assertTrue(rag.evidence_gate([{}], [('A', .30)], [{}])[0])
        self.assertFalse(rag.evidence_gate([{}], [('A', .299)], [{'dense_rank': 1}])[0])
        self.assertEqual(rag.evidence_gate([{}], [('A', .1)], [{'dense_rank': 1, 'bm25_rank': 1}]),
                         (True, 'retriever_agreement'))

    def test_real_langchain_bm25_and_retrieval_prompt(self):
        assets = assets_fixture()
        from langchain_core.documents import Document
        self.assertIsInstance(assets.bm25.docs[0], Document)
        embedder = Mock()
        embedder.encode.return_value = np.array([[1, 0, 0]], dtype='float32')
        result = rag.retrieve(assets, embedder, 'SOLAR')
        self.assertTrue(result['accepted'])
        self.assertEqual(result['sparse'][0][0], assets.chunk_ids[0])
        embedder.encode.assert_called_once_with(['SOLAR'], convert_to_numpy=True, normalize_embeddings=True)
        contexts = rag.retrieve_context(assets, embedder, 'SOLAR')
        self.assertTrue(contexts[0].startswith('[SOURCE 1]'))
        prompt = rag.build_evidence_prompt('SOLAR', contexts)
        self.assertIn('File: 0.txt', prompt)
        self.assertIn('Section: Energy', prompt)
        self.assertIn('only the supplied evidence', prompt)
        self.assertEqual(rag.build_evidence_prompt('Original question?', []), 'Original question?')

    def test_rejected_retrieval_and_bad_query_vectors(self):
        assets = assets_fixture()
        embedder = Mock()
        embedder.encode.return_value = np.array([[-1, -1, -1]], dtype='float32')
        result = rag.retrieve(assets, embedder, 'unmatchedword')
        self.assertFalse(result['accepted'])
        self.assertEqual(result['sources'], [])
        self.assertEqual(rag.retrieve_context(assets, embedder, 'unmatchedword'), [])
        for vector in ([[0, 0, 0]], [[float('nan'), 0, 0]], [[1, 0]]):
            embedder.encode.return_value = vector
            with self.assertRaises(ValueError):
                rag.retrieve(assets, embedder, 'q')
        with self.assertRaisesRegex(ValueError, 'Legacy'):
            rag.retrieve(assets.index, embedder, 'q')

    def test_parent_dedup_budget_and_missing_parent(self):
        assets = assets_fixture()
        first = assets.children[0]
        assets.children.append({**first, 'chunk_id': 'duplicate-parent-child'})
        selected = [{'chunk_id': first['chunk_id']}, {'chunk_id': 'duplicate-parent-child'}]
        sources = rag.expand_parents(selected, assets)
        self.assertEqual(len(sources), 1)
        assets.parents[first['parent_id']]['text'] = 'x' * 15000
        sources = rag.expand_parents(selected, assets)
        self.assertEqual(sum(len(s['text']) for s in sources), 12000)
        assets.parents.clear()
        self.assertEqual(rag.expand_parents(selected, assets), [])

    def test_asset_roundtrip_legacy_preservation_and_tamper_rejection(self):
        assets = assets_fixture()
        meta = pd.DataFrame([{'doc_id': str(i), 'filename': f'{i}.txt', 'file_hash': 'hash'} for i in range(4)])
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            folder = Path(temp)
            (folder / 'index.faiss').write_bytes(b'legacy')
            with self.assertRaisesRegex(ValueError, 'not built'):
                rag.load_rag_assets(folder)
            rag.publish_assets(folder, assets.index, assets.children, list(assets.parents.values()), assets.embeddings, meta)
            loaded = rag.load_rag_assets(folder)
            np.testing.assert_array_equal(loaded.embeddings, assets.embeddings)
            self.assertEqual(loaded.children, assets.children)
            self.assertEqual((folder / 'index.faiss').read_bytes(), b'legacy')
            path = rag.active_generation(folder) / 'documents.jsonl'
            path.write_text(path.read_text() + '\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'changed'):
                rag.load_rag_assets(folder)

    def test_vector_mapping_and_parent_validation(self):
        assets = assets_fixture()
        parents = list(assets.parents.values())
        for children, parent_records, vectors, ids in [
            (assets.children, parents, assets.embeddings * 2, assets.chunk_ids),
            (assets.children, parents, assets.embeddings, list(reversed(assets.chunk_ids))),
            (assets.children, [], assets.embeddings, assets.chunk_ids),
            (assets.children, parents, assets.embeddings[::-1], assets.chunk_ids),
        ]:
            with self.assertRaises(ValueError):
                rag.validate_assets(assets.index, children, parent_records, vectors, ids)
        wrong = faiss.IndexFlatL2(3)
        wrong.add(assets.embeddings)
        with self.assertRaisesRegex(ValueError, 'IndexFlatIP'):
            rag.validate_assets(wrong, assets.children, parents, assets.embeddings, assets.chunk_ids)

    def test_publication_failure_keeps_previous_generation(self):
        assets = assets_fixture()
        meta = pd.DataFrame([{'doc_id': str(i), 'filename': f'{i}.txt', 'file_hash': 'hash'} for i in range(4)])
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            folder = Path(temp)
            rag.publish_assets(folder, assets.index, assets.children, list(assets.parents.values()), assets.embeddings, meta)
            previous = (folder / 'advanced' / 'CURRENT.json').read_bytes()
            replace = Path.replace
            def fail_pointer(path, target):
                if Path(target).name == 'CURRENT.json':
                    raise OSError('simulated failure')
                return replace(path, target)
            with patch.object(Path, 'replace', fail_pointer):
                with self.assertRaises(OSError):
                    rag.publish_assets(folder, assets.index, assets.children, list(assets.parents.values()), assets.embeddings, meta)
            self.assertEqual((folder / 'advanced' / 'CURRENT.json').read_bytes(), previous)
            self.assertEqual(rag.load_rag_assets(folder).chunk_ids, assets.chunk_ids)

    def test_incremental_addition_modification_and_extraction_failure(self):
        encoded = []
        class Embedder:
            def encode(self, texts, **kwargs):
                encoded.append(list(texts))
                return np.array([[1, (len(text) % 11 + 1) / 20, .2] for text in texts], dtype='float32')
        tracker = Mock()
        tracker.stop.return_value = .001
        factory = Mock(return_value=tracker)
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            base = Path(temp)
            corpus, index = base / 'corpus', base / 'index'
            corpus.mkdir()
            (corpus / 'a.txt').write_text('# Solar\nSolar power produces electricity.', encoding='utf-8')
            def build(number, extractor=lambda p, converter: p.read_text(encoding='utf-8')):
                rag.index_corpus(corpus, index, base / f'report{number}', {'.txt'}, extractor,
                                 Mock, lambda name: Embedder(), factory, lambda path: {'energy_consumed': .003})
            with contextlib.redirect_stdout(io.StringIO()):
                build(1)
                initial = rag.load_rag_assets(index)
                build(2)
                self.assertEqual(len(encoded), 1)  # No changes, no model/tracker work.
                (corpus / 'b.txt').write_text('# Water\nWater powers irrigation.', encoding='utf-8')
                build(3)
                added = rag.load_rag_assets(index)
                np.testing.assert_array_equal(added.embeddings[0], initial.embeddings[0])
                self.assertTrue(all('Solar' not in text for text in encoded[1]))
                (corpus / 'a.txt').write_text('# Solar\nSolar panels have changed.', encoding='utf-8')
                (corpus / 'c.txt').write_text('# Trees\nTrees store carbon.', encoding='utf-8')
                build(4)
                updated = rag.load_rag_assets(index)
                self.assertEqual({c['filename'] for c in updated.children}, {'a.txt', 'b.txt', 'c.txt'})
                old_water = next(i for i, c in enumerate(added.children) if c['filename'] == 'b.txt')
                new_water = next(i for i, c in enumerate(updated.children) if c['filename'] == 'b.txt')
                np.testing.assert_array_equal(added.embeddings[old_water], updated.embeddings[new_water])
                summary = pd.read_csv(base / 'report4' / 'summary.csv').iloc[0]
                self.assertEqual(summary['num_new_docs'], 1)
                self.assertEqual(summary['num_modified_docs'], 1)
                self.assertEqual(summary['index_type'], 'IndexFlatIP')
                previous = (index / 'advanced' / 'CURRENT.json').read_bytes()
                (corpus / 'a.txt').write_text('Changed again.', encoding='utf-8')
                with self.assertRaises(ValueError):
                    build(5, extractor=lambda p, converter: None)
                self.assertEqual((index / 'advanced' / 'CURRENT.json').read_bytes(), previous)
                self.assertEqual(tracker.stop.call_count, 7)

    def test_rag_runner_uses_real_hybrid_context_with_unchanged_metrics(self):
        assets = assets_fixture()
        prompts, trackers = [], []
        def query(prompt, model):
            prompts.append((prompt, model))
            return 'Answer [SOURCE 1]'
        process = load_functions('run.py', query, trackers)
        process.__wrapped__.__globals__['retrieve_context'] = rag.retrieve_context
        process.__wrapped__.__globals__['build_augmented_prompt'] = rag.build_evidence_prompt
        embedder = Mock()
        embedder.encode.return_value = np.array([[1, 0, 0]], dtype='float32')
        with tempfile.TemporaryDirectory(dir=ROOT) as temp, contextlib.chdir(temp), contextlib.redirect_stdout(io.StringIO()):
            process(pd.DataFrame([{'Question ID': 1, 'Question': 'Solar electricity?'}]), True,
                    assets, [c['text'] for c in assets.children], embedder, 'selected-model')
            folder = next(Path('emissions_reports').iterdir())
            answers = pd.read_csv(folder / 'answers.csv')
            self.assertEqual(len(trackers), 2)
            self.assertTrue(all(t.stops == 1 for t in trackers))
            self.assertAlmostEqual(answers.iloc[0]['energy_kwh'], .006)
            self.assertAlmostEqual(answers.iloc[0]['total_emissions_kg'], .002)
            self.assertIn('[SOURCE 1]', answers.iloc[0]['chunks'])
            self.assertIn('Section: Energy', prompts[-1][0])
            self.assertTrue(all(model == 'selected-model' for _, model in prompts))
            self.assertEqual(json.loads((folder / 'run_status.json').read_text())['status'], 'completed')
            from rag_reporting import read_details
            details = read_details(folder)
            self.assertEqual(len(details['groups']), 2)
            retrieval = details['queries'][0]['retrieval']
            self.assertEqual(retrieval['response_mode'], 'rag')
            self.assertTrue(all(value >= 0 for value in retrieval['timings'].values()))
            self.assertEqual(details['groups'][0]['cpu_energy_kwh'], .001)
            self.assertEqual(details['groups'][0]['gpu_energy_kwh'], 0)
            self.assertTrue((folder / 'modular_emissions.xlsx').is_file())

    def test_cancelled_advanced_indexing_keeps_previous_assets(self):
        assets = assets_fixture()
        meta = pd.DataFrame([{'doc_id': str(i), 'filename': f'{i}.txt', 'file_hash': 'hash'} for i in range(4)])
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            base = Path(temp)
            corpus, folder, output = base / 'corpus', base / 'index', base / 'report'
            corpus.mkdir()
            (corpus / 'new.txt').write_text('# New\nSome new evidence.', encoding='utf-8')
            rag.publish_assets(folder, assets.index, assets.children, list(assets.parents.values()), assets.embeddings, meta)
            previous = (folder / 'advanced' / 'CURRENT.json').read_bytes()
            tracker = Tracker(output, 'emissions.csv', 'indexing')
            embedder = Mock()
            embedder.encode.side_effect = KeyboardInterrupt
            @cancellation.cancellable_run
            def build():
                rag.index_corpus(corpus, folder, output, {'.txt'}, lambda p, c: p.read_text(),
                                 Mock, lambda name: embedder, lambda *a, **k: tracker, lambda path: {})
            with contextlib.redirect_stdout(io.StringIO()):
                build()
            self.assertEqual(tracker.stops, 2)
            self.assertEqual((folder / 'advanced' / 'CURRENT.json').read_bytes(), previous)
            status = json.loads((output / 'run_status.json').read_text())
            self.assertEqual(status['status'], 'cancelled')
            self.assertFalse(status['index_published'])


if __name__ == '__main__':
    unittest.main()
