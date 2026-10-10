"""Explicit follow-up chains, isolated chat history, and auditable context limits."""
import json
import re
from pathlib import Path

FOLLOW_UP = re.compile(r'follow[- ]up\s+(\d+)', re.I)
CONFIG_PATH = Path(__file__).resolve().with_name('conversation_config.json')


class MissingHistoryError(RuntimeError):
    pass


def annotate_chains(frame):
    """Numbered contiguous Follow-Up 1..N sequences define chains within a sheet."""
    frame = frame.copy()
    frame['conversation_chain'] = ''
    frame['conversation_turn'] = 0
    frame['conversation_prerequisites'] = [[] for _ in range(len(frame))]
    if 'question_type' not in frame:
        return frame
    for topic in dict.fromkeys(frame['topic_sheet']):
        chain, previous_turn, prerequisites = None, 0, []
        for index, row in frame[frame['topic_sheet'] == topic].iterrows():
            label = str(row['question_type']).strip()
            match = FOLLOW_UP.fullmatch(label)
            if match is None:
                if re.search(r'follow[- ]?up', label, re.I):
                    raise ValueError(f'{topic}: ambiguous follow-up label {label!r}. Use numbered Follow-Up 1..N chains.')
                chain, previous_turn, prerequisites = None, 0, []
                continue
            turn = int(match.group(1))
            qid = str(row['Question ID'])
            if turn == 1:
                chain, previous_turn, prerequisites = f'{topic}::{qid}', 0, []
            if chain is None or turn != previous_turn + 1:
                raise ValueError(f'{topic}: follow-up {qid} is missing a contiguous earlier turn. Start each chain at Follow-Up 1.')
            frame.at[index, 'conversation_chain'] = chain
            frame.at[index, 'conversation_turn'] = turn
            frame.at[index, 'conversation_prerequisites'] = list(prerequisites)
            prerequisites.append(qid)
            previous_turn = turn
    return frame


def execution_units(frame, indexes):
    """Each independent question is one unit; each chain is one ordered unit."""
    units, seen = [], set()
    for index in indexes:
        chain = frame.loc[index].get('conversation_chain', '')
        if chain:
            if chain in seen:
                continue
            seen.add(chain)
            members = [i for i in indexes if frame.loc[i].get('conversation_chain', '') == chain]
            units.append(sorted(members, key=lambda i: int(frame.loc[i]['conversation_turn'])))
        else:
            units.append([index])
    return units


def validate_execution(frame):
    seen = set()
    for _, row in frame.iterrows():
        prerequisites = row.get('conversation_prerequisites', [])
        missing = [qid for qid in prerequisites if qid not in seen]
        if missing:
            raise ValueError(f"Follow-up {row.get('Question ID')} requires earlier turns: {', '.join(missing)}. "
                             'Include them and preserve chain order before starting the run.')
        if row.get('conversation_chain'):
            seen.add(str(row['Question ID']))


class ConversationMemory:
    """Fresh per-run history. Only successful turns are retained; unrelated chains never mix."""
    def __init__(self, frame, folder, model):
        if 'question_type' in frame and any(FOLLOW_UP.fullmatch(str(v).strip()) for v in frame['question_type']) and 'conversation_chain' not in frame:
            raise ValueError('Follow-up rows need chain metadata. Load them through question_order.load_bank/select_questions.')
        validate_execution(frame)
        self.folder, self.model = Path(folder), model
        self.history, self.records, self.options = {}, [], None
        self.retrieval_records = {}
        self.settings = {}
        if any(frame.get('conversation_chain', [])):
            import requests
            self.settings = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))
            for key in ('num_ctx', 'num_predict', 'template_margin_tokens'):
                if type(self.settings.get(key)) is not int or self.settings[key] <= 0:
                    raise ValueError(f'Invalid conversation configuration: {key}')
            if self.settings.get('overflow_policy') != 'reject_without_truncation' or self.settings.get('budget_method') != 'conservative_utf8_bytes':
                raise ValueError('Unsupported conversation context policy.')
            try:
                response = requests.post('http://127.0.0.1:11434/api/show', json={'model': model}, timeout=10)
                response.raise_for_status()
                info = response.json().get('model_info', {})
                architecture_key = str(info.get('general.architecture', '')) + '.context_length'
                exposed = [value for key, value in info.items() if key.endswith('.context_length')]
                limit = info.get(architecture_key, exposed[0] if len(exposed) == 1 else None)
                if limit is None or int(limit) <= 0:
                    raise ValueError('model context capacity was not exposed')
                limit = int(limit)
            except (requests.RequestException, ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f'Cannot verify follow-up context capacity for {model}: {exc}') from exc
            context = min(self.settings['num_ctx'], limit)
            self.options = {'num_ctx': context, 'num_predict': self.settings['num_predict']}
            self.byte_budget = context - self.settings['num_predict'] - self.settings['template_margin_tokens']
            if self.byte_budget <= 0:
                raise ValueError('Conversation answer reserve and template margin leave no input budget.')
        self.save()

    def save(self):
        from cancellation import protect_cleanup
        with protect_cleanup(defer_interrupt=True):
            path = self.folder / 'conversation_context.json'
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps({'schema_version': 1, 'model': self.model,
                'settings': self.settings, 'effective_options': self.options,
                'history_policy': 'successful_turns_within_chain; no summarization or truncation',
                'requests': self.records}, ensure_ascii=False, indent=2), encoding='utf-8')
            temporary.replace(path)

    def prior(self, row):
        chain = row.get('conversation_chain', '')
        if not chain:
            return []
        turns = self.history.get(chain, [])
        expected = row.get('conversation_prerequisites', [])
        if [t['question_id'] for t in turns] != expected:
            raise MissingHistoryError('Follow-up blocked: an earlier answer failed or is unavailable. No history was invented.')
        return turns

    def retrieval_query(self, row, query, embedder=None):
        """History-aware retrieval without another LLM call or query rewriting."""
        if not row.get('conversation_chain'):
            return query
        turns = self.prior(row)
        # Preserve the current question first; MiniLM has a smaller input limit than the generator.
        # Original opening question provides an explicit subject; recent turns provide referents.
        retrieval_query = (query + '\n\nConversation subject: ' + turns[0]['question'] + '\n\nRecent conversation:\n' + '\n'.join(
            f"User: {t['question']}\nAssistant: {t['answer']}" for t in turns[-2:])) if turns else query
        maximum = getattr(embedder, 'max_seq_length', None)
        tokenizer = getattr(embedder, 'tokenizer', None)
        detail = {'query': retrieval_query, 'history_question_ids': [t['question_id'] for t in turns[-2:]],
                  'opening_question_id': turns[0]['question_id'] if turns else str(row['Question ID']), 'encoder_limit': maximum if type(maximum) is int else None,
                  'encoder_truncated': None}
        if type(maximum) is int and maximum > 0 and tokenizer is not None:
            ids = tokenizer.encode(retrieval_query, add_special_tokens=True, truncation=False)
            detail['encoder_truncated'] = len(ids) > maximum
            if detail['encoder_truncated']:
                ids = tokenizer.encode(retrieval_query, add_special_tokens=True, truncation=True, max_length=maximum)
                retrieval_query = tokenizer.decode(ids, skip_special_tokens=True)
                detail['query'] = retrieval_query
        self.retrieval_records[str(row['Question ID'])] = detail
        return retrieval_query

    def generate(self, query_function, row, prompt, model):
        chain = row.get('conversation_chain', '')
        if not chain:
            return query_function(prompt, model=model)
        record = {'question_id': str(row['Question ID']), 'chain_id': chain,
                  'turn': int(row['conversation_turn']), 'execution_position': int(row['execution_position']),
                  'status': 'started', 'preceding_question_ids': [], 'messages': [], 'input_utf8_bytes': None,
                  'truncated': False, 'summarized': False, 'context_options': self.options}
        if str(row['Question ID']) in self.retrieval_records:
            record['retrieval_context'] = self.retrieval_records[str(row['Question ID'])]
        self.records.append(record)
        try:
            turns = self.prior(row)
            messages = []
            for turn in turns:
                messages.extend([{'role': 'user', 'content': turn['question']},
                                 {'role': 'assistant', 'content': turn['answer']}])
            messages.append({'role': 'user', 'content': prompt})
            record.update(preceding_question_ids=[t['question_id'] for t in turns], messages=messages,
                          input_utf8_bytes=sum(len(m['content'].encode('utf-8')) for m in messages))
            budget = self.byte_budget - 64 * len(messages)
            record['input_byte_budget'] = budget
            if record['input_utf8_bytes'] > budget:
                raise RuntimeError(f"Follow-up input exceeds the conservative context budget ({budget} UTF-8 bytes). "
                                   'History was not truncated. Adjust conversation_config.json or use a larger-context model.')
            answer = query_function(prompt, model=model, messages=messages, options=self.options)
        except BaseException as exc:
            record.update(status='cancelled' if isinstance(exc, KeyboardInterrupt) else 'error', error=str(exc))
            self.save()
            raise
        record.update(status='ok', answer=answer)
        self.save()
        self.history.setdefault(chain, []).append({'question_id': str(row['Question ID']),
                                                   'question': str(row['Question']), 'answer': answer})
        return answer
