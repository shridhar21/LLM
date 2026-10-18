"""Recount historical saved text with the local Ollama llama.cpp vocabulary.

No inference, model downloads, or edits to Ollama-reported counts.
Example: python reconstruct_tokens.py emissions_reports/SV emissions_reports/SN emissions_reports/SC --write
"""
import argparse
import csv
import ctypes as C
import hashlib
import io
import json
import os
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# ABI checked against the installed binary's exact source commit, not guessed.
# https://github.com/ggml-org/llama.cpp/blob/161755f29/include/llama.h
SUPPORTED_COMMIT = '161755f29'
TEMPLATES = {
    'llama3.2:3b': 'sha256:966de95ca8a62200913e3f8bfbf84c8494536f1b94b49166851e76644e966396',
    'gemma3:4b': 'sha256:e0a42594d802e5d31cdc786deb4823edb8adff66094d49de8fffe976d753e348',
    'qwen3:4b': 'sha256:2d54db2b9bb29ce7db54fea63a891f5859603813c555b1f88b5e0994652897f9',
}
FIELDS = ['reconstructed_input_tokens', 'reconstructed_output_tokens', 'token_count_method']


class ModelParams(C.Structure):
    _fields_ = [('devices',C.c_void_p),('tensor_buft_overrides',C.c_void_p),
                ('n_gpu_layers',C.c_int32),('split_mode',C.c_int),('load_mode',C.c_int),
                ('lazy_mode',C.c_int),('main_gpu',C.c_int32),('tensor_split',C.c_void_p),
                ('progress_callback',C.c_void_p),('progress_callback_user_data',C.c_void_p),
                ('kv_overrides',C.c_void_p),('vocab_only',C.c_bool),('check_tensors',C.c_bool),
                ('use_extra_bufts',C.c_bool),('no_host',C.c_bool),('no_alloc',C.c_bool),('load_mtp',C.c_bool)]


class NativeTokenizer:
    """Vocabulary-only load: no context, weights, sampling, or generated tokens."""
    def __init__(self, library_dir):
        library_dir = Path(library_dir).resolve()
        version = subprocess.run([str(library_dir/'llama-server.exe'),'--version'],
                                 capture_output=True,text=True,check=True,timeout=15,
                                 creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if SUPPORTED_COMMIT not in version.stdout+version.stderr or C.sizeof(ModelParams)!=80:
            raise ValueError('Unsupported native library ABI. Update bindings against its exact llama.h before running.')
        self.dll_path = library_dir/'libllama.dll'
        self.dll_directory = os.add_dll_directory(str(library_dir))
        self.lib = C.CDLL(str(self.dll_path))
        for name,restype,args in [
            ('llama_model_default_params',ModelParams,[]),
            ('llama_model_load_from_file',C.c_void_p,[C.c_char_p,ModelParams]),
            ('llama_model_get_vocab',C.c_void_p,[C.c_void_p]),
            ('llama_model_free',None,[C.c_void_p]),
            ('llama_tokenize',C.c_int32,[C.c_void_p,C.c_char_p,C.c_int32,C.POINTER(C.c_int32),C.c_int32,C.c_bool,C.c_bool]),
            ('llama_vocab_n_tokens',C.c_int32,[C.c_void_p]),
            ('llama_vocab_get_add_bos',C.c_bool,[C.c_void_p]),
            ('llama_vocab_get_add_eos',C.c_bool,[C.c_void_p]),
        ]:
            function=getattr(self.lib,name);function.restype=restype;function.argtypes=args
        self.log_callback=C.CFUNCTYPE(None,C.c_int,C.c_char_p,C.c_void_p)(lambda *_:None)
        self.lib.llama_log_set.argtypes=[C.c_void_p,C.c_void_p]
        self.lib.llama_log_set(self.log_callback,None)
        self.models={}
        self.version=version.stdout.strip() or version.stderr.strip()

    def load(self,model,path):
        params=self.lib.llama_model_default_params()
        params.vocab_only=True;params.n_gpu_layers=0
        pointer=self.lib.llama_model_load_from_file(str(path).encode('utf-8'),params)
        if not pointer:raise ValueError(f'Cannot load vocabulary for {model}')
        vocab=self.lib.llama_model_get_vocab(pointer)
        self.models[model]=(pointer,vocab)
        return dict(vocabulary_size=self.lib.llama_vocab_n_tokens(vocab),
                    add_bos=bool(self.lib.llama_vocab_get_add_bos(vocab)),
                    add_eos=bool(self.lib.llama_vocab_get_add_eos(vocab)))

    def count(self,model,text,input_prompt=False):
        raw=text.encode('utf-8');vocab=self.models[model][1]
        required=self.lib.llama_tokenize(vocab,raw,len(raw),None,0,input_prompt,True)
        if required==0:return 0
        if required>0 or required==-2147483648:raise ValueError('Unexpected tokenizer allocation result')
        tokens=(C.c_int32*(-required))()
        size=self.lib.llama_tokenize(vocab,raw,len(raw),tokens,len(tokens),input_prompt,True)
        if size<0:raise ValueError('Tokenization failed')
        return size

    def close(self):
        for pointer,_ in self.models.values():self.lib.llama_model_free(pointer)
        self.models.clear()


def model_assets(root,model):
    if model not in TEMPLATES:raise ValueError(f'Unvalidated model/template: {model}')
    name,tag=model.split(':',1)
    manifest_path=root/'manifests'/'registry.ollama.ai'/'library'/name/tag
    manifest_bytes=manifest_path.read_bytes();manifest=json.loads(manifest_bytes)
    layers={entry['mediaType'].rsplit('.',1)[-1]:entry['digest'] for entry in manifest['layers']}
    if layers.get('template')!=TEMPLATES[model]:
        raise ValueError(f'{model}: local template changed; reconstruction renderer needs validation')
    template_path=root/'blobs'/layers['template'].replace(':','-')
    if hashlib.sha256(template_path.read_bytes()).hexdigest()!=layers['template'].split(':')[1]:
        raise ValueError('Template file disagrees with its manifest digest')
    return root/'blobs'/layers['model'].replace(':','-'),dict(
        model=model,declared_model_blob_digest=layers['model'],template_digest=layers['template'],
        current_manifest_digest='sha256:'+hashlib.sha256(manifest_bytes).hexdigest())


def render_prompt(model,messages):
    """Validated text-only branches of the three local Ollama Go templates."""
    if not messages or messages[-1]['role']!='user':raise ValueError('Saved prompt must end in the current user question')
    if any(m.get('role') not in ('user','assistant') or not isinstance(m.get('content'),str) for m in messages):
        raise ValueError('Unsupported saved message format')
    if model=='llama3.2:3b':
        result='<|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\n\n<|eot_id|>'
        for message in messages:
            result+=f'<|start_header_id|>{message["role"]}<|end_header_id|>\n\n{message["content"]}<|eot_id|>'
        return result+'<|start_header_id|>assistant<|end_header_id|>\n\n'
    if model=='gemma3:4b':
        return ''.join(f'<start_of_turn>{"model" if m["role"]=="assistant" else "user"}\n{m["content"]}<end_of_turn>\n' for m in messages)+'<start_of_turn>model\n'
    if model=='qwen3:4b':
        return ''.join(f'<|im_start|>{m["role"]}\n{m["content"]}<|im_end|>\n' for m in messages)+'<|im_start|>assistant\n<think>\n'
    raise ValueError('Unsupported prompt template')


def reconstruct_rows(rows,context,counter):
    saved={str(r.get('question_id')):r for r in context.get('requests',[])}
    rebuilt=[]
    for original in rows:
        if 'retrieval_latency_s' in original:
            raise ValueError('RAG inputs require saved augmented prompts; this historical reconstruction supports normal LLM reports only')
        row=dict(original);model=row['model_name']
        row.update(reconstructed_input_tokens='',reconstructed_output_tokens='',token_count_method='')
        if row.get('status')!='ok':
            row['token_count_method']='unavailable: failed/unknown request'
            rebuilt.append(row);continue
        row['reconstructed_output_tokens']=counter(model,row.get('answer',''),False)
        request=saved.get(str(row.get('question_id')))
        followup=bool(re.fullmatch(r'Follow[ -]?Up\s*\d+',row.get('question_type',''),re.IGNORECASE))
        messages=request.get('messages') if request else None
        if request:
            if (context.get('model')!=model or request.get('status')!='ok' or
                request.get('answer')!=row.get('answer') or
                str(request.get('execution_position'))!=str(row.get('execution_position')) or
                not messages or messages[-1].get('content')!=row['question']):
                row['token_count_method']='output: saved answer; input unavailable: saved history mismatch'
                rebuilt.append(row);continue
        elif followup:
            row['token_count_method']='output: saved answer; input unavailable: missing follow-up history'
            rebuilt.append(row);continue
        else:messages=[{'role':'user','content':row['question']}]
        row['reconstructed_input_tokens']=counter(model,render_prompt(model,messages),True)
        row['token_count_method']=('reconstructed: local native tokenizer; '+
            ('saved follow-up messages' if request else 'saved question')+' + local Ollama template; saved answer only')
        rebuilt.append(row)
    return rebuilt


def read_csv(path):
    original=path.read_bytes();reader=csv.DictReader(io.StringIO(original.decode('utf-8-sig'),newline=''))
    fields=list(reader.fieldnames or []);rows=list(reader)
    if 'question' not in fields or len(fields)!=len(set(fields)) or any(None in r for r in rows):
        raise ValueError(f'Invalid CSV: {path}')
    return original,fields,rows


def publish(path,original,fields,old,new,provenance):
    backup=path.with_name(path.name+'.before_token_reconstruction.bak')
    if backup.exists():raise FileExistsError(f'Existing reconstruction backup; refusing to overwrite: {backup}')
    if path.read_bytes()!=original:raise ValueError(f'Source changed during reconstruction: {path}')
    with backup.open('xb') as handle:handle.write(original)
    columns=fields+[key for key in FIELDS if key not in fields]
    # SV may not yet have the future API-count columns. Leave these explicitly empty.
    columns += [key for key in ('input_tokens','output_tokens') if key not in columns]
    temporary=path.with_name(path.name+'.reconstruct.tmp')
    with temporary.open('x',encoding='utf-8-sig' if original.startswith(b'\xef\xbb\xbf') else 'utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader();writer.writerows(new)
    _,_,verified=read_csv(temporary)
    if len(old)!=len(verified) or any({k:r[k] for k in fields}!=before for before,r in zip(old,verified)):
        raise ValueError('Original values failed verification; source retained')
    if any(str(r[k])!=str(v.get(k,'')) for r,v in zip(verified,new) for k in FIELDS):
        raise ValueError('Reconstructed values failed verification; source retained')
    sidecar=path.parent/'token_reconstruction.json'
    with sidecar.open('x',encoding='utf-8') as handle:json.dump(provenance,handle,indent=2,ensure_ascii=False)
    os.replace(temporary,path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folders',nargs='+',type=Path)
    parser.add_argument('--models',type=Path,default=Path(os.environ.get('OLLAMA_MODELS',str(Path.home()/'.ollama/models'))))
    parser.add_argument('--library-dir',type=Path,default=Path(os.environ.get('LOCALAPPDATA',''))/'Programs/Ollama/lib/ollama')
    parser.add_argument('--write',action='store_true')
    args=parser.parse_args();paths=[]
    for root in args.folders:
        root=root.resolve()
        if not root.is_dir():parser.error(f'Folder missing: {root}')
        for p in sorted(root.rglob('answers.csv')):
            if not p.resolve().is_relative_to(root):raise ValueError('Report resolves outside the selected folder')
            paths.append(p)
    if len(set(p.resolve() for p in paths))!=len(paths):raise ValueError('Overlapping input folders')
    inputs=[(p,*read_csv(p)) for p in paths]
    if any(set(FIELDS)&set(fields) for _,_,fields,_ in inputs):raise ValueError('Reconstruction columns already exist; sources left unchanged')
    for path,_,_,_ in inputs:
        if path.with_name(path.name+'.before_token_reconstruction.bak').exists() or (path.parent/'token_reconstruction.json').exists():
            raise ValueError(f'Prior reconstruction assets exist: {path}')
    native=NativeTokenizer(args.library_dir);assets={};counts=Counter()
    try:
        for model in sorted({r['model_name'] for _,_,_,rows in inputs for r in rows}):
            blob,assets[model]=model_assets(args.models,model)
            assets[model].update(native.load(model,blob))
            print('Vocabulary loaded:',model,assets[model]['vocabulary_size'],flush=True)
        for path,original,fields,rows in inputs:
            ctx_path=path.parent/'conversation_context.json'
            ctx=json.loads(ctx_path.read_text(encoding='utf-8')) if ctx_path.exists() else {}
            new=reconstruct_rows(rows,ctx,native.count)
            summary=dict(rows=len(rows),inputs=sum(r['reconstructed_input_tokens']!='' for r in new),
                         outputs=sum(r['reconstructed_output_tokens']!='' for r in new),
                         unavailable=sum(r['reconstructed_output_tokens']=='' for r in new))
            provenance=dict(schema_version=1,created_at=datetime.now(timezone.utc).isoformat(),
                source_sha256=hashlib.sha256(original).hexdigest(),
                source_context_sha256=hashlib.sha256(ctx_path.read_bytes()).hexdigest() if ctx_path.exists() else None,
                native_library_commit=SUPPORTED_COMMIT,native_library_sha256=hashlib.sha256(native.dll_path.read_bytes()).hexdigest(),
                models=[assets[m] for m in sorted({r['model_name'] for r in rows})],summary=summary,
                scope='Saved text retokenization, not recovered historical Ollama usage counters',
                input_policy='Current local Ollama template and tokenizer; saved successful follow-up messages or standalone question; configured BOS/EOS inserted',
                output_policy='Saved answer text only; no automatic BOS/EOS; control tokens in saved text parsed',
                limitations=['Historical model/template digests were not recorded and cannot be verified across machines.',
                             'Removed whitespace, omitted thinking, hidden termination tokens and generation token boundaries cannot be recovered.',
                             'Failed/unknown requests remain unavailable; original API-count fields are not populated.'])
            if args.write:publish(path,original,fields,rows,new,provenance)
            counts.update(summary)
            print(path.parent.name,summary,flush=True)
    finally:native.close()
    print('WRITTEN' if args.write else 'DRY RUN',len(inputs),'runs',dict(counts))


if __name__=='__main__':main()
