"""Shared workbook selection and reproducible execution ordering for both pipelines."""
import json
import random
import re
import secrets
from pathlib import Path

import pandas as pd
from conversation import annotate_chains, execution_units

BANK = Path(__file__).resolve().parent / 'questionbank.xlsx'
MODES = [
    ('original', 'Keep original order', False, False, False, False),
    ('questions', 'Shuffle questions within each question type; keep topic sheets and question types in original order', False, False, True, False),
    ('question_types', 'Shuffle question types within each topic sheet; keep topic sheet order and question order within each question type', False, True, False, False),
    ('question_types_questions', 'Shuffle question types and questions within each question type; keep topic sheet order', False, True, True, False),
    ('topics', 'Shuffle topic sheets only', True, False, False, False),
    ('topics_questions', 'Shuffle topic sheets and questions within each question type; keep question type order', True, False, True, False),
    ('topics_question_types', 'Shuffle topic sheets and question types; keep question order within each question type', True, True, False, False),
    ('all_levels', 'Shuffle topic sheets, question types, and questions within each question type', True, True, True, False),
    ('within_topics', 'Keep topic sheet order; freely shuffle all questions within each topic sheet', False, False, False, True),
    ('topics_within_topics', 'Shuffle topic sheets; freely shuffle all questions within each topic sheet', True, False, False, True),
    ('global', 'Freely shuffle all questions across all topic sheets', False, False, False, False),
]
RANGE_LABELS = {
    'original': 'Keep original order',
    'questions': 'Shuffle questions within each question type; keep question types in original order',
    'question_types': 'Shuffle question types; keep questions within each question type in original order',
    'question_types_questions': 'Shuffle question types and the questions within each question type',
    'within_topics': 'Freely shuffle all selected questions regardless of question type',
}


def load_bank(path=BANK):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f'Question workbook not found: {path}')
    sheets = pd.read_excel(path, sheet_name=None)
    frames = []
    for sheet, frame in sheets.items():
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        if 'question text' not in frame:
            raise ValueError(f'{sheet}: missing Question Text column')
        frame = frame[frame['question text'].notna()].copy()
        if frame.empty:
            continue
        for column in ('question number', 'question type', 'domain'):
            if column not in frame or frame[column].isna().any():
                raise ValueError(f'{sheet}: missing {column} values')
        numbers = pd.to_numeric(frame['question number'], errors='coerce')
        if numbers.isna().any() or (numbers <= 0).any() or (numbers % 1 != 0).any() or numbers.duplicated().any():
            raise ValueError(f'{sheet}: question numbers must be unique positive integers')
        frame['question_number'] = numbers.astype(int)
        frame['topic_sheet'] = sheet
        frame['question_type'] = frame['question type'].astype(str).str.strip()
        frame['domain'] = frame['domain'].astype(str)
        frame['Question'] = frame['question text'].astype(str)
        frame['Question ID'] = sheet + ':' + frame['question_number'].astype(str)
        frames.append(frame[['Question ID', 'Question', 'topic_sheet', 'question_number', 'question_type', 'domain']])
    if not frames:
        raise ValueError('The workbook contains no questions')
    return annotate_chains(pd.concat(frames, ignore_index=True))


def group_name(value):
    return 'Follow-Up' if re.fullmatch(r'follow[- ]up\s+\d+', str(value), re.I) else str(value)


def grouped(frame):
    result = {}
    for index, row in frame.iterrows():
        result.setdefault(group_name(row['question_type']), []).append(index)
    return list(result.values())


def free_order(frame, indexes, rng):
    """Randomly choose an available question; unlock the next turn of its chain."""
    units = execution_units(frame, indexes)
    if not any(len(unit) > 1 for unit in units):
        order = list(indexes)
        rng.shuffle(order)
        return order
    order = []
    while units:
        picked = rng.randrange(len(units))
        order.append(units[picked].pop(0))
        if not units[picked]:
            units.pop(picked)
    return order


def followup_policy(mode):
    if mode in ('within_topics', 'topics_within_topics', 'global'):
        return 'preserve_followup_turn_order; allow_interleaving'
    return 'preserve_followup_question_type_order; no_internal_shuffle'


def followup_note(mode):
    if mode in ('within_topics', 'topics_within_topics', 'global'):
        return 'follow-up turns stay ordered; other questions may run between them'
    return 'follow-up questions keep their original order within their question type'


def arrange(frame, mode='original', seed=None):
    frame = frame.reset_index(drop=True).copy()
    spec = next((m for m in MODES if m[0] == mode), None)
    if spec is None:
        raise ValueError('Unknown ordering mode')
    seed = seed if seed is not None else secrets.randbits(32)
    rng = random.Random(seed)
    _, description, shuffle_topics, shuffle_types, shuffle_questions, freely = spec
    topics = list(dict.fromkeys(frame['topic_sheet']))
    if shuffle_topics:
        rng.shuffle(topics)
    indexes = []
    for topic in topics:
        subset = frame[frame['topic_sheet'] == topic]
        groups = grouped(subset)
        if freely:
            order = free_order(frame, list(subset.index), rng)
        elif mode == 'original' or mode == 'topics':
            order = list(subset.index)
        else:
            if shuffle_types:
                rng.shuffle(groups)
            if shuffle_questions:
                for j, group in enumerate(groups):
                    if group_name(frame.loc[group[0], 'question_type']) == 'Follow-Up':
                        continue
                    units = execution_units(frame, group)
                    rng.shuffle(units)
                    groups[j] = [i for unit in units for i in unit]
            order = [i for group in groups for i in group]
        indexes.extend(order)
    if mode == 'global':
        indexes = free_order(frame, list(frame.index), rng)
    result = frame.loc[indexes].reset_index(drop=True)
    result['execution_position'] = range(1, len(result) + 1)
    result['ordering_mode'] = mode
    result['shuffle_seed'] = seed if mode != 'original' else None
    result.attrs['ordering'] = {'mode': mode, 'description': description, 'seed': seed if mode != 'original' else None}
    if any(frame.get('conversation_chain', [])):
        result.attrs['ordering']['conversation_order_policy'] = followup_policy(mode)
        result.attrs['ordering']['description'] += ' (' + followup_note(mode) + ')'
    return result


def choose_integer(prompt, allowed):
    while True:
        try:
            value = int(input(prompt))
            if value in allowed:
                return value
        except ValueError:
            pass
        print('Please enter one of the displayed valid numbers.')


def range_options(frame):
    """Offer distinct ordering operations for the selected questions."""
    if frame['topic_sheet'].nunique() == 1:
        groups = grouped(frame.reset_index(drop=True))
        many_types = len(groups) > 1
        normalized = frame.reset_index(drop=True)
        many_questions = any(len(g) > 1 and group_name(normalized.loc[g[0], 'question_type']) != 'Follow-Up'
                             for g in groups)
        options = [MODES[0]]
        if len(execution_units(frame.reset_index(drop=True), list(range(len(frame))))) > 1:
            if many_types and many_questions:
                options += [MODES[1], MODES[2], MODES[3], MODES[8]]
            elif many_types:
                options += [MODES[2]]
                if any(len(unit) > 1 for unit in execution_units(normalized, list(normalized.index))):
                    options += [MODES[8]]
            else:
                options += [MODES[8]]
        return [(spec[0], RANGE_LABELS[spec[0]], *spec[2:]) for spec in options]

    # Describe movable blocks, rather than comparing random outcomes for one seed.
    # Singleton blocks and adjacent fixed blocks have no independent ordering effect.
    def block(children, shuffled=False):
        children = list(children)
        if not shuffled:
            children = [item for child in children
                        for item in (child[1] if child[0] == 'fixed' else [child])]
        if len(children) == 1:
            return children[0]
        return ('shuffle', tuple(sorted(children, key=repr))) if shuffled else ('fixed', tuple(children))

    frame = frame.reset_index(drop=True)
    leaves = {i: ('question', i) for i in frame.index}
    def units(indexes):
        return [block(leaves[i] for i in unit) for unit in execution_units(frame, list(indexes))]
    def free_signature(indexes):
        ordered_units = execution_units(frame, list(indexes))
        if len(ordered_units) <= 1:
            return block(units(indexes))
        constraints = tuple(sorted((a, b) for unit in ordered_units for a, b in zip(unit, unit[1:])))
        if not constraints:
            return block(units(indexes), True)
        return ('interleave', tuple(sorted(indexes)), constraints)
    options, seen = [], set()
    for spec in MODES:
        mode, _, shuffle_topics, shuffle_types, shuffle_questions, freely = spec
        topics = []
        for topic in dict.fromkeys(frame['topic_sheet']):
            subset = frame[frame['topic_sheet'] == topic]
            if freely:
                topics.append(free_signature(subset.index))
            elif mode in ('original', 'topics'):
                topics.append(block(units(subset.index)))
            else:
                groups = [block(units(group), shuffle_questions and
                                group_name(frame.loc[group[0], 'question_type']) != 'Follow-Up')
                          for group in grouped(subset)]
                topics.append(block(groups, shuffle_types))
        signature = free_signature(frame.index) if mode == 'global' else block(topics, shuffle_topics)
        if signature not in seen:
            seen.add(signature)
            options.append(spec)
    return options


def select_questions(choice):
    frame = load_bank()
    bank = frame.copy()
    selection = None
    if choice in ('1', '3'):
        topics = list(dict.fromkeys(frame['topic_sheet']))
        print('\nChoose a topic sheet:')
        for i, topic in enumerate(topics, 1):
            print(f'{i}. {topic}')
        if choice == '3':
            print(f'{len(topics)+1}. Free range across topic sheets')
        picked = choose_integer('Topic sheet number: ', range(1, len(topics)+(2 if choice == '3' else 1)))
        if picked == len(topics)+1:
            frame['workbook_position'] = range(1, len(frame)+1)
            bank = frame.copy()
            print('\nWorkbook serial | Topic sheet | Sheet serial | Question type')
            for _, row in frame.iterrows():
                print(f"{row['workbook_position']:>15} | {row['topic_sheet']} | {row['question_number']} | {row['question_type']}")
            print('Workbook serials follow sheet order and question row order. Both endpoints are included.')
            numbers = set(frame['workbook_position'])
            start = choose_integer('Starting workbook serial number: ', numbers)
            end = choose_integer('Ending workbook serial number (inclusive): ', {n for n in numbers if n >= start})
            frame = frame[frame['workbook_position'].between(start, end)].copy()
            selection = {'kind': 'free_range', 'start': start, 'end': end, 'numbering': 'workbook_position'}
        else:
            topic = topics[picked-1]
            frame = frame[frame['topic_sheet'] == topic].copy()
            print('\nSerial number | Question type')
            for _, row in frame.iterrows():
                print(f"{row['question_number']:>13} | {row['question_type']}")
            numbers = set(frame['question_number'])
            start = choose_integer('Starting serial number: ' if choice == '3' else 'Question serial number: ', numbers)
            end = choose_integer('Ending serial number (inclusive): ', {n for n in numbers if n >= start}) if choice == '3' else start
            frame = frame[frame['question_number'].between(start, end)].copy()
    required = {qid for prerequisites in frame.get('conversation_prerequisites', []) for qid in prerequisites}
    missing = required - set(frame['Question ID'])
    if missing:
        print('\nThis selection omits earlier turns needed by a follow-up chain:')
        print(', '.join(bank[bank['Question ID'].isin(missing)]['Question ID']))
        print('1. Include and execute the earlier turns (their costs are also recorded)')
        print('2. Cancel this selection')
        if choose_integer('Choose (1-2): ', (1, 2)) == 2:
            raise ValueError('Selection cancelled: follow-ups require their earlier questions and answers.')
        frame = bank[bank['Question ID'].isin(set(frame['Question ID']) | missing)].copy()
        selection = dict(selection or {}, prerequisites_added=sorted(missing))
    if choice == '1':
        result = arrange(frame)
        if selection:
            result.attrs['ordering']['selection'] = selection
        return result
    options = range_options(frame) if choice == '3' else MODES
    print('\nChoose question execution order:')
    print('Follow-up questions keep their original turn order. Free shuffles may place other questions between turns; within-question-type shuffles leave follow-ups in order.')
    for i, spec in enumerate(options, 1):
        print(f'{i}. {spec[1]}')
    selected = options[choose_integer(f'Enter your choice (1-{len(options)}): ', range(1,len(options)+1))-1]
    result = arrange(frame, selected[0])
    result.attrs['ordering']['description'] = selected[1]
    if any(frame.get('conversation_chain', [])):
        result.attrs['ordering']['description'] += ' (' + followup_note(selected[0]) + ')'
    if selection:
        result.attrs['ordering']['selection'] = selection
    print(f"Selected {len(result)} questions. Shuffle seed: {result.attrs['ordering']['seed']}")
    return result


def identity(row, position):
    result = {key: row.get(key, default) for key, default in
            [('topic_sheet', ''), ('question_number', ''), ('question_type', ''), ('domain', ''),
             ('execution_position', position), ('ordering_mode', 'original'), ('shuffle_seed', None)]}
    if 'workbook_position' in row:
        result['workbook_position'] = row['workbook_position']
    return result


def save_order(frame, folder):
    columns = [c for c in ('Question ID', 'topic_sheet', 'question_number', 'question_type', 'execution_position', 'workbook_position', 'conversation_chain', 'conversation_turn') if c in frame]
    metadata = dict(frame.attrs.get('ordering', {'mode': 'original', 'seed': None}))
    metadata['questions'] = frame[columns].to_dict(orient='records')
    (Path(folder) / 'question_order.json').write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
