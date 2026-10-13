"""Score a single civil case with the models saved by scripts/build_dashboard.py.

Raw case fields are rebuilt into the notebook's features (scaled numerics + one-hot
categoricals -> PCA) using the transformers exported by scripts/build_predictor.py.
Labels follow the FJC Civil Codebook (1988 forward).
"""
from functools import lru_cache
from pathlib import Path
import warnings

import numpy as np

ARTIFACTS = Path(__file__).resolve().parent/'artifacts'
PREPROCESS = ARTIFACTS/'predictor/preprocess.joblib'

DISTRICTS = {
    '00': 'Maine', '01': 'Massachusetts', '02': 'New Hampshire', '03': 'Rhode Island', '04': 'Puerto Rico',
    '05': 'Connecticut', '06': 'New York - Northern', '07': 'New York - Eastern', '08': 'New York - Southern',
    '09': 'New York - Western', '10': 'Vermont', '11': 'Delaware', '12': 'New Jersey', '13': 'Pennsylvania - Eastern',
    '14': 'Pennsylvania - Middle', '15': 'Pennsylvania - Western', '16': 'Maryland', '17': 'North Carolina - Eastern',
    '18': 'North Carolina - Middle', '19': 'North Carolina - Western', '20': 'South Carolina', '22': 'Virginia - Eastern',
    '23': 'Virginia - Western', '24': 'West Virginia - Northern', '25': 'West Virginia - Southern',
    '26': 'Alabama - Northern', '27': 'Alabama - Middle', '28': 'Alabama - Southern', '29': 'Florida - Northern',
    '3A': 'Florida - Middle', '3C': 'Florida - Southern', '3E': 'Georgia - Northern', '3G': 'Georgia - Middle',
    '3J': 'Georgia - Southern', '3L': 'Louisiana - Eastern', '3N': 'Louisiana - Middle', '36': 'Louisiana - Western',
    '37': 'Mississippi - Northern', '38': 'Mississippi - Southern', '39': 'Texas - Northern', '40': 'Texas - Eastern',
    '41': 'Texas - Southern', '42': 'Texas - Western', '43': 'Kentucky - Eastern', '44': 'Kentucky - Western',
    '45': 'Michigan - Eastern', '46': 'Michigan - Western', '47': 'Ohio - Northern', '48': 'Ohio - Southern',
    '49': 'Tennessee - Eastern', '50': 'Tennessee - Middle', '51': 'Tennessee - Western', '52': 'Illinois - Northern',
    '53': 'Illinois - Central', '54': 'Illinois - Southern', '55': 'Indiana - Northern', '56': 'Indiana - Southern',
    '57': 'Wisconsin - Eastern', '58': 'Wisconsin - Western', '60': 'Arkansas - Eastern', '61': 'Arkansas - Western',
    '62': 'Iowa - Northern', '63': 'Iowa - Southern', '64': 'Minnesota', '65': 'Missouri - Eastern',
    '66': 'Missouri - Western', '67': 'Nebraska', '68': 'North Dakota', '69': 'South Dakota', '7-': 'Alaska',
    '70': 'Arizona', '71': 'California - Northern', '72': 'California - Eastern', '73': 'California - Central',
    '74': 'California - Southern', '75': 'Hawaii', '76': 'Idaho', '77': 'Montana', '78': 'Nevada', '79': 'Oregon',
    '80': 'Washington - Eastern', '81': 'Washington - Western', '82': 'Colorado', '83': 'Kansas', '84': 'New Mexico',
    '85': 'Oklahoma - Northern', '86': 'Oklahoma - Eastern', '87': 'Oklahoma - Western', '88': 'Utah', '89': 'Wyoming',
    '90': 'District of Columbia', '91': 'Virgin Islands', '93': 'Guam', '94': 'Northern Mariana Islands'}
# Every district maps to exactly one circuit in cv23 (checked against the raw file).
CIRCUIT_OF = {
    **dict.fromkeys(['00', '01', '02', '03', '04'], 1), **dict.fromkeys(['05', '06', '07', '08', '09', '10'], 2),
    **dict.fromkeys(['11', '12', '13', '14', '15', '91'], 3), **dict.fromkeys(['16', '17', '18', '19', '20', '22', '23', '24', '25'], 4),
    **dict.fromkeys(['3L', '3N', '36', '37', '38', '39', '40', '41', '42'], 5),
    **dict.fromkeys(['43', '44', '45', '46', '47', '48', '49', '50', '51'], 6),
    **dict.fromkeys(['52', '53', '54', '55', '56', '57', '58'], 7),
    **dict.fromkeys(['60', '61', '62', '63', '64', '65', '66', '67', '68', '69'], 8),
    **dict.fromkeys(['7-', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '93', '94'], 9),
    **dict.fromkeys(['82', '83', '84', '85', '86', '87', '88', '89'], 10),
    **dict.fromkeys(['26', '27', '28', '29', '3A', '3C', '3E', '3G', '3J'], 11), '90': 0}
CIRCUITS = {0: 'District of Columbia Circuit', 1: 'First', 2: 'Second', 3: 'Third', 4: 'Fourth', 5: 'Fifth', 6: 'Sixth',
            7: 'Seventh', 8: 'Eighth', 9: 'Ninth', 10: 'Tenth', 11: 'Eleventh'}
ORIGINS = {1: 'Original proceeding', 2: 'Removed from state court', 3: 'Remanded from court of appeals',
           4: 'Reinstated / reopened', 5: 'Transferred from another district (28 USC 1404)',
           6: 'Multidistrict litigation transfer (28 USC 1407)', 8: 'Second reopen', 9: 'Third reopen',
           10: 'Fourth reopen', 11: 'Fifth reopen', 12: 'Sixth reopen', 13: 'Multidistrict litigation originating here'}
JURISDICTIONS = {1: 'U.S. government plaintiff', 2: 'U.S. government defendant', 3: 'Federal question',
                 4: 'Diversity of citizenship', 5: 'Local question'}
NOS_GROUPS = ['Civil Rights', 'Contract', 'Immigration', 'Intellectual Property', 'Labor', 'Other',
              'Prisoner Petitions', 'Social Security', 'Torts - Personal Injury']
JURY = {'N': 'Neither party', 'P': 'Plaintiff', 'D': 'Defendant', 'B': 'Both parties'}
PROSE = {0: 'No pro se parties', 1: 'Pro se plaintiff(s)', 2: 'Pro se defendant(s)', 3: 'Pro se on both sides'}
CITIZENSHIP = {1: 'Citizen of this State', 2: 'Citizen of another State', 3: 'Citizen or subject of a foreign country',
               4: 'Incorporated or principal place of business in this State',
               5: 'Incorporated and principal place of business in another State', 6: 'Foreign nation'}
PROCPROG = {1: 'Before issue joined: no court action', 2: 'Before issue joined: order entered',
            11: 'Before issue joined: hearing held', 12: 'Before issue joined: order decided',
            3: 'After issue joined: no court action', 4: 'After issue joined: judgment on motion',
            5: 'After issue joined: pretrial conference held', 6: 'During court trial', 7: 'During jury trial',
            8: 'After court trial', 9: 'After jury trial', 10: 'After issue joined: other',
            13: 'Request for trial de novo after arbitration'}
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
          'November', 'December']

# The most common value of each field in cv23; used as the starting case and as the
# reference point when explaining which inputs moved a prediction.
TYPICAL = dict(district='08', origin=1, juris=3, nos_group='Civil Rights', jury='P', demanded=0.0,
               plaintiff_cit=None, defendant_cit=None, classact=0, prose=0, ifp=0, procprog=1, filing_month=6)

FIELDS = [
    ('district', 'Court district', 'Court & filing'), ('filing_month', 'Filing month', 'Court & filing'),
    ('origin', 'How the case arrived', 'Court & filing'), ('juris', 'Jurisdiction basis', 'Case type'),
    ('nos_group', 'Nature of suit', 'Case type'), ('classact', 'Class action', 'Case type'),
    ('citizenship', 'Party citizenship (diversity only)', 'Case type'), ('demanded', 'Amount demanded', 'Parties & stakes'),
    ('jury', 'Jury demand', 'Parties & stakes'), ('prose', 'Self-represented parties', 'Parties & stakes'),
    ('ifp', 'In forma pauperis', 'Parties & stakes'), ('procprog', 'Procedural progress', 'Progress')]


def schema():
    opts = lambda d: [[k, v] for k, v in d.items()]
    return dict(
        typical=TYPICAL, fields=[dict(key=k, label=l, group=g) for k, l, g in FIELDS],
        options=dict(
            district=[[k, f'{v} ({k})'] for k, v in sorted(DISTRICTS.items(), key=lambda kv: kv[1])],
            circuit_of=CIRCUIT_OF, circuits=CIRCUITS, origin=opts(ORIGINS), juris=opts(JURISDICTIONS),
            nos_group=[[g, g] for g in NOS_GROUPS], jury=opts(JURY), prose=opts(PROSE), citizenship=opts(CITIZENSHIP),
            procprog=opts(PROCPROG), filing_month=[[i + 1, m] for i, m in enumerate(MONTHS)]))


def _pick(value, allowed, name, cast):
    try:
        value = cast(value)
    except (TypeError, ValueError):
        raise ValueError(f'Invalid {name}.') from None
    if value not in allowed:
        raise ValueError(f'Unknown {name}: {value}.')
    return value


def normalise(raw):
    """Validate user input and return the 16 raw features the preprocessing expects."""
    case = {**TYPICAL, **{k: v for k, v in raw.items() if v is not None and v != ''}}
    district = _pick(case['district'], DISTRICTS, 'district', str)
    juris = _pick(case['juris'], JURISDICTIONS, 'jurisdiction', int)
    month = _pick(case['filing_month'], range(1, 13), 'filing month', int)
    try:
        demanded = float(case['demanded'])
    except (TypeError, ValueError):
        raise ValueError('Amount demanded must be a number (in $ thousands).') from None
    if not 0 <= demanded <= 9999:
        raise ValueError('Amount demanded must be between 0 and 9999 ($ thousands; 9999 means over $10M).')
    if 'residenc' in raw:  # raw two-digit code, used when replaying dataset rows
        residenc = int(raw['residenc'])
    elif juris == 4 and case.get('plaintiff_cit') and case.get('defendant_cit'):
        residenc = 10*_pick(case['plaintiff_cit'], CITIZENSHIP, 'plaintiff citizenship', int) + \
            _pick(case['defendant_cit'], CITIZENSHIP, 'defendant citizenship', int)
    else:
        residenc = 0  # the notebook maps "not a diversity case" (-8) to 0
    return dict(
        CIRCUIT=float(CIRCUIT_OF[district]), DISTRICT=district,
        ORIGIN=float(_pick(case['origin'], ORIGINS, 'origin', int)), JURIS=float(juris),
        NOS_GROUP=_pick(case['nos_group'], NOS_GROUPS, 'nature of suit', str),
        JURY=_pick(case['jury'], JURY, 'jury demand', str),
        DEMANDED=demanded, RESIDENC=float(residenc), CLASSACT=float(_pick(case['classact'], (0, 1), 'class action', int)),
        PROSE=float(_pick(case['prose'], PROSE, 'pro se', int)), IFP_FLAG=float(_pick(case['ifp'], (0, 1), 'IFP', int)),
        IS_DIVERSITY_CASE=float(juris == 4), PROCPROG=float(_pick(case['procprog'], PROCPROG, 'procedural progress', int)),
        FILING_MONTH=float(month), FILING_QUARTER=float((month - 1)//3 + 1), HIGH_STAKES=float(demanded >= 100))


@lru_cache(maxsize=1)
def _preprocess():
    import joblib
    if not PREPROCESS.is_file():
        raise FileNotFoundError('Prediction preprocessing is missing. Run python scripts/build_predictor.py.')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # transformers were pickled by a neighbouring sklearn patch release
        return joblib.load(PREPROCESS)


def features(*cases):
    """Raw feature dicts -> PCA feature matrix, exactly as in notebooks/preprocessing.ipynb."""
    import pandas as pd
    p = _preprocess()
    frame = pd.DataFrame(list(cases))
    numeric = pd.DataFrame(p['scaler'].transform(frame[p['numeric_cols']]), columns=p['numeric_cols'])
    encoded = pd.DataFrame(p['encoder'].transform(frame[p['categorical_cols']]),
                           columns=p['encoder'].get_feature_names_out(p['categorical_cols']))
    return p['pca'].transform(pd.concat([numeric, encoded], axis=1))


def notebook_models():
    """Tuned models saved by the notebooks (scored on the full test set by build_predictor.py)."""
    import json
    path = ARTIFACTS/'predictor/notebook_models.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else []


@lru_cache(maxsize=32)
def load_model(run_id, model_id):
    import joblib
    saved = next((m for m in notebook_models() if m['id'] == model_id), None)
    if saved:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            local = ARTIFACTS/saved['path']  # bundled copy (Streamlit app) before the project's models/ folder
            return joblib.load(local if local.is_file() else ARTIFACTS.parents[2]/saved['path'])
    path = ARTIFACTS/'runs'/run_id/'models'/(model_id + '.joblib')
    if not path.is_file():
        raise FileNotFoundError(f'Saved model {model_id} is missing for run {run_id}.')
    return joblib.load(path)


def _perturbations(raw):
    """One counterfactual per input group: that input reset to the typical case."""
    for key, label, _ in FIELDS:
        if key == 'citizenship':
            if raw.get('plaintiff_cit') is None and raw.get('defendant_cit') is None:
                continue
            changed = {**raw, 'plaintiff_cit': None, 'defendant_cit': None}
        else:
            if str(raw.get(key, TYPICAL[key])) == str(TYPICAL[key]):
                continue
            changed = {**raw, key: TYPICAL[key]}
        if key == 'juris' and TYPICAL['juris'] != 4:
            changed = {**changed, 'plaintiff_cit': None, 'defendant_cit': None}
        yield label, normalise(changed)


def predict(task, run_id, models, chosen_id, raw):
    """Score the case with every model of the task; explain the chosen one."""
    case = normalise(raw)
    extras = list(_perturbations(raw))
    X = features(case, *[c for _, c in extras])
    chosen = next(m for m in models if m['id'] == chosen_id)
    rows = []
    for m in models:
        model = load_model(run_id, m['id'])
        if task == 'classification':
            proba = model.predict_proba(X[:1])[0]
            classes = [str(c) for c in model.classes_]
            best = int(np.argmax(proba))
            rows.append(dict(id=m['id'], name=m['name'], prediction=classes[best], confidence=float(proba[best]),
                             probabilities=dict(zip(classes, map(float, proba))), test=m.get('test', {})))
        else:
            rows.append(dict(id=m['id'], name=m['name'], prediction=float(model.predict(X[:1])[0]), test=m.get('test', {})))
    model = load_model(run_id, chosen_id)
    main = next(r for r in rows if r['id'] == chosen_id)
    effects = []
    if extras:
        if task == 'classification':
            label = main['prediction']
            index = [str(c) for c in model.classes_].index(label)
            others = model.predict_proba(X[1:])[:, index]
            base = main['probabilities'][label]
            effects = [dict(input=l, effect=float(base - o)) for (l, _), o in zip(extras, others)]
        else:
            others = model.predict(X[1:])
            effects = [dict(input=l, effect=float(main['prediction'] - o)) for (l, _), o in zip(extras, others)]
        effects.sort(key=lambda e: -abs(e['effect']))
    return dict(task=task, model=dict(id=chosen['id'], name=chosen['name'], test=chosen.get('test', {})),
                result=main, all_models=rows, effects=effects,
                case=dict(circuit=CIRCUITS[int(case['CIRCUIT'])], high_stakes=bool(case['HIGH_STAKES']),
                          diversity=bool(case['IS_DIVERSITY_CASE']), residence_code=int(case['RESIDENC'])))
