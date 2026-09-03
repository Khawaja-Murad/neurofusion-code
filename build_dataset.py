"""
NeuroFusion Dataset Builder
Processes voice-dictated radiology transcripts into structured clinical data.
Maps ASR-garbled case numbers, corrects medical terminology, writes proper reports,
and extracts structured columns matching the master file format.
"""

import pandas as pd
import json
import os
import re
from numbers_parser import Document

###############################################################################
# 1. CASE NUMBER MAPPING (manually verified from sequential analysis)
###############################################################################

CASE_MAP = {
    0: "TR184", 1: "TR185", 2: "TR186", 3: "TR187", 4: "TR188",
    5: "TR189", 6: "TR190", 7: "TR191", 8: "TR192", 9: "TR193",
    10: "TR194", 11: "TR194_DUP", 12: "TR195", 13: "TR196", 14: "TR197",
    15: "TR198", 16: "TR199", 17: "TR200", 18: "TR201", 19: "TR202",
    20: "TR203", 21: "TR204", 22: "TR205", 23: "TR206", 24: "TR207",
    25: "TR208", 26: "TR209", 27: "TR210", 28: "TR211", 29: "TR212",
    30: "TR213", 31: "TR214", 32: "TR215", 33: "TR216", 34: "TR217",
    35: "TR218", 36: "TR219", 37: "TR220", 38: "TR221", 39: "TR222",
    40: "TR223", 41: "TR224", 42: "TR225", 43: "TR226", 44: "TR227",
    45: "TR228", 46: "TR229", 47: "TR230", 48: "TR231",
    49: "TR301", 50: "TR302", 51: "TR303", 52: "TR304", 53: "TR305",
    54: "TR306", 55: "TR307", 56: "TR309", 57: "TR308", 58: "TR310",
    59: "TR311", 60: "TR312", 61: "TR313", 62: "TR314", 63: "TR315",
    64: "TR316", 65: "SKIP", 66: "TR317", 67: "TR318", 68: "TR319",
    69: "TR320", 70: "TR182", 71: "TR321", 72: "TR322", 73: "TR323",
    74: "TR324", 75: "TR325", 76: "TR326", 77: "TR327", 78: "TR328",
    79: "TR329", 80: "TR183", 81: "TR330", 82: "TR331", 83: "TR181",
    84: "TR310_DUP",
}

###############################################################################
# 2. ASR MEDICAL TERMINOLOGY CORRECTIONS
###############################################################################

def correct_asr_medical(text):
    """Apply systematic ASR corrections for medical terminology.
    Order matters - more specific patterns before general ones."""
    
    if not text or text == 'NaN':
        return text
    
    corrections = [
        # Remove case number prefix (will be handled by TR mapping)
        (r'^(?:Case|It\'s|It is|In case|On case|Here is|Age|Gage|CHS\.?\s*)\s*[\d\s.\-aTorand]+,?\s*', ''),
        
        # Diagnosis/findings phrasing
        (r'\bThis instance is a strip of\b', 'Findings suggestive of'),
        (r'\bThis is likely suggestive of\b', 'Findings suggestive of'),
        (r'\bIt is a ring like a suggestive of\b', 'Findings suggestive of'),
        (r'\bfinding like a suggestive of\b', 'findings suggestive of'),
        (r'\bFinding like the suggestive of\b', 'Findings suggestive of'),
        (r'\bFindings like the suggestive of\b', 'Findings suggestive of'),
        (r'\bfindings like the suggestive of\b', 'findings suggestive of'),
        (r'\bfinding like a suggestive\b', 'findings suggestive'),
        (r'\bfinding that this is a tip of\b', 'findings suggestive of'),
        (r'\bFinding glycemic tissue of\b', 'Findings suggestive of'),
        (r'\bfinding glycemic tissue of\b', 'findings suggestive of'),
        (r'\bLacking differential\b', 'Likely differential'),
        (r'\blacking differential\b', 'likely differential'),
        (r'\blikely the frontal diagnosis\b', 'likely differential diagnosis'),
        (r'\bPridings\b', 'Findings'),
        (r'\bpridings\b', 'findings'),
        (r'\bAppearances are suggestive of\b', 'Findings suggestive of'),
        (r'\bappearances are suggestive of\b', 'findings suggestive of'),
        (r'\bfinding the suggestive of\b', 'findings suggestive of'),
        (r'\bfinding suggestive of\b', 'findings suggestive of'),
        (r'\bfindings are consistent with\b', 'findings consistent with'),
        (r'\bFindings are consistent with\b', 'Findings consistent with'),
        (r'\bfindings consistent with findings\b', 'findings consistent with'),
        
        # Compound ASR errors (fix before individual word corrections)
        (r'\bperitoneal demoral edema\b', 'peritumoral edema'),
        (r'\bperitoneal relatively uniform\b', 'peripheral relatively uniform'),
        (r'\bcerebral celsiogari\b', 'falx cerebri'),
        (r'\bcystic meds or nepsis\b', 'cystic metastasis or abscess'),
        (r'\bcystic mets or abscess\b', 'cystic metastasis or abscess'),
        (r'\bor nepsis\b', 'or abscess'),
        (r'\bthe blastoma multi-form\b', 'glioblastoma multiforme'),
        (r'\bblastoma multi-form\b', 'glioblastoma multiforme'),
        (r'\bblastoma multiform\b', 'glioblastoma multiforme'),
        (r'\bmulti-form\b', 'multiforme'),
        (r'\bmultiform\b', 'multiforme'),
        (r'\bglioblastoma multiforme\b', 'glioblastoma multiforme'),  # normalize
        (r'\bGlioblastoma Multiforme\b', 'glioblastoma multiforme'),
        (r'\bglauoma\b', 'glioma'),
        (r'\bglomerular\b', 'glioma'),
        (r'\bhigh grade glioblastoma multiform\b', 'high-grade glioblastoma multiforme'),
        (r'\bNSOL\b', 'SOL'),
        (r'\bremind the placement\b', 'with displacement'),
        (r'\bwith displacement of the left lateral ventricle\b', 'with effacement of the left lateral ventricle'),
        (r'\bmidline shape\b', 'midline shift'),
        (r'\bmidline-shaped\b', 'midline shift'),
        (r'\bmidline shift finding\b', 'midline shift. Finding'),
        (r'\bthe ligand has very mild marked\b', 'the lesion has marked'),
        (r'\bstanding from\b', 'extending from'),
        (r'\bis standing\b', 'is extending'),
        (r'\badhesion brain\b', 'adjacent brain'),
        (r'\bvarietal lobe\b', 'parietal lobe'),
        (r'\bvarietal\b', 'parietal'),
        (r'\blima\b(?=\s+(?:in|with|causing|and))', 'edema'),
        (r'\bperilesional lima\b', 'perilesional edema'),
        (r'\bperitumoral lima\b', 'peritumoral edema'),
        (r'\bslash\b', '/'),
        (r'\bperitermoid\b', 'peritumoral'),
        (r'\bperitermoidal\b', 'peritumoral'),
        (r'\bperiolational\b', 'perilesional'),
        (r'\bHere is\b(?=\s+(?:a|an|large|small))', 'There is'),
        (r'^Here is\b', 'There is'),
        (r'\bsemiovel\b', 'semiovale'),
        (r'\bcentrum semiovale\b', 'centrum semiovale'),
        (r'\brightal\b', 'parietal'),
        (r'\bperisyllium fissure\b', 'perisylvian fissure'),
        
        # Edema terms (most specific first)
        (r'\bperitoneal\s+edema\b', 'peritumoral edema'),
        (r'\bperitermoral\b', 'peritumoral'),
        (r'\bperitemoral\b', 'peritumoral'),
        (r'\bperitomoral\b', 'peritumoral'),
        (r'\bperitimodal\b', 'peritumoral'),
        (r'\bperithemoral\b', 'peritumoral'),
        (r'\bperitremoral\b', 'peritumoral'),
        (r'\bperitermal\b', 'peritumoral'),
        (r'\bperi tumoral\b', 'peritumoral'),
        (r'\bperitoneal\b', 'peritumoral'),  # in edema context
        (r'\bperilegional\b', 'perilesional'),
        (r'\bperilaginal\b', 'perilesional'),
        (r'\bperilational\b', 'perilesional'),
        (r'\bperilisional\b', 'perilesional'),
        (r'\bperilegial\b', 'perilesional'),
        (r'\bnepsis\b', 'abscess'),
        (r'\bparietimoral\b', 'peritumoral'),
        (r'\bentear\b', 'anterior'),
        (r'\binterior\b(?=\s+horn)', 'anterior'),
        (r'\babinine\b', 'benign'),
        
        # Lesion terms
        (r'\bligands\b', 'lesions'),
        (r'\bligand\b', 'lesion'),
        (r'\blegions\b', 'lesions'),
        (r'\blegion\b', 'lesion'),
        (r'\blean\b(?!\s+(?:on|forward|back|against|towards|toward))', 'lesion'),  # avoid "lean on"
        (r'\bleons?\b', 'lesion'),
        (r'\bmedian\b(?=\s+(?:in|on|with|showing))', 'lesion'),  # only in lesion context
        (r'\blayer\b(?=\s+in\s+the)', 'lesion'),
        (r'\bthe end\b(?=\s+in\s+the)', 'lesion'),
        
        # Anatomy terms
        (r'\blobee\b', 'lobe'),
        (r'\bfox\b(?=\s+(?:region|margin|cerebri))', 'falx'),
        (r'\bgrope\b', 'groove'),
        (r'\bvortex\b', 'vertex'),
        (r'\bbasement\b(?=\s+of\s+the)', 'effacement'),
        (r'\bpara-central\b', 'paracentral'),
        (r'\bgrounded\b(?=\s+cystic)', 'rounded'),
        (r'\bfront of parietal\b', 'frontoparietal'),
        (r'\bfrontal parietotemporal\b', 'frontoparieto-temporal'),
        (r'\bfrontal parietal temporal\b', 'fronto-parieto-temporal'),
        (r'\bfrontal parietal\b', 'fronto-parietal'),
        (r'\bparietal temporal\b', 'parieto-temporal'),
        (r'\bparietal occipital\b', 'parieto-occipital'),
        (r'\btemporal parietal\b', 'temporo-parietal'),
        (r'\bparietotemporal\b', 'parieto-temporal'),
        (r'\bfrontoparietal\b', 'fronto-parietal'),
        (r'\btemporo-parietal\b', 'temporo-parietal'),
        
        # Enhancement terms
        (r'\blaryngal enhancement\b', 'ring enhancement'),
        (r'\blaryngeal enhancement\b', 'ring enhancement'),
        (r'\bhydrogenous\b', 'heterogeneous'),
        (r'\bilipine enhancement\b', 'irregular enhancement'),
        (r'\bheterogenous\b', 'heterogeneous'),
        (r'\bdroidinous\b', 'heterogeneous'),
        (r'\bsensuality\b', 'centrality'),
        
        # Other medical terms
        (r'\bcariform pattern\b', 'cruciform pattern'),
        (r'\bintracist\b', 'intracystic'),
        (r'\bintracistic\b', 'intracystic'),
        (r'\binter-cystic\b', 'intracystic'),
        (r'\bcallousum\b', 'callosum'),
        (r'\bcorpus callosum\b', 'corpus callosum'),
        (r'\bat least\s*T2\b', 'at least 2'),
        (r'\bat leasT2\b', 'at least 2'),
        
        # Clean up
        (r'\bof a local\b', 'of a focal'),
        (r'\ba hollow-shaped\b', 'an oval-shaped'),
        (r'\ba oval-shaped\b', 'an oval-shaped'),
        (r'\bcystic enhancing area\b', 'cystic/enhancing lesion'),
        (r'\benhancing area\b(?=\s+(?:in|on|along|within))', 'enhancing lesion'),
        (r'\bcystic area\b(?=\s+in\s+the)', 'cystic lesion'),
        (r'\boval shaped area\b', 'oval-shaped lesion'),
        (r'\boval-shaped area\b', 'oval-shaped lesion'),
        (r'\bcystic peripherally ring enhancing area\b', 'cystic peripherally ring-enhancing lesion'),
    ]
    
    result = text
    for pattern, replacement in corrections:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    return result.strip()


###############################################################################
# 3. STRUCTURED DATA EXTRACTION
###############################################################################

def extract_hemisphere(text):
    """Extract hemisphere from report text."""
    text_lower = text.lower()
    
    has_left = bool(re.search(r'\bleft\b', text_lower))
    has_right = bool(re.search(r'\bright\b', text_lower))
    has_both = bool(re.search(r'\bboth\b|\bbilateral\b|\bmidline\b.*\bboth\b', text_lower))
    
    # Check for bilateral indicators
    if has_both or (has_left and has_right and 'extending' in text_lower and 'both' in text_lower):
        return 'both'
    
    # Look for hemisphere in first mention of lobe
    first_lobe = re.search(r'(left|right)\s+\w*\s*(?:lobe|parietal|frontal|temporal|occipital|cerebral|cerebellar)', text_lower)
    if first_lobe:
        return first_lobe.group(1)
    
    if has_right and not has_left:
        return 'right'
    if has_left and not has_right:
        return 'left'
    if has_right:
        return 'right'
    if has_left:
        return 'left'
    return ''

def extract_tumor_location(text):
    """Extract tumor location following master file vocabulary."""
    text_lower = text.lower()
    
    # Standardized location patterns (matching master file vocabulary)
    location_patterns = [
        (r'fronto-parieto-temporal\s+lobe', 'fronto-parieto-temporal lobes'),
        (r'fronto-parietal\s+lobe', 'fronto-parietal lobes'),
        (r'fronto-parietal\s+and\s+temporal\s+lobe', 'fronto-parieto-temporal lobes'),
        (r'frontal\s+parietal\s+and\s+temporal\s+lobe', 'fronto-parieto-temporal lobes'),
        (r'frontal\s+and\s+parietal\s+lobe', 'fronto-parietal lobes'),
        (r'parieto-temporal\s+(?:lobe|region)', 'parieto-temporal lobe'),
        (r'temporo-parietal\s+(?:lobe|region)', 'temporo-parietal lobe'),
        (r'parieto-occipital\s+(?:lobe|region)', 'parieto-occipital lobe'),
        (r'occipito-parietal\s+(?:lobe|region)', 'parieto-occipital lobe'),
        (r'posterior\s+parietal\s+(?:lobe|region)', 'parietal lobe'),
        (r'parietal\s+(?:lobe|region|convexity)', 'parietal lobe'),
        (r'frontal\s+(?:lobe|region)', 'frontal lobe'),
        (r'temporal\s+lobe', 'temporal lobe'),
        (r'occipital\s+lobe', 'occipital lobe'),
        (r'cerebellar\s+hemisphere', 'cerebellar hemisphere'),
        (r'cerebral\s+hemispheres?', 'cerebral hemisphere'),
        (r'vermis', 'vermis'),
        (r'centrum\s+semiovale', 'centrum semiovale'),
        (r'midbrain', 'midbrain'),
        (r'basal\s+ganglia', 'basal ganglia'),
        (r'corpus\s+callosum', 'corpus callosum'),
        (r'perisylvian', 'parietal lobe'),
    ]
    
    for pattern, name in location_patterns:
        if re.search(pattern, text_lower):
            return name
    
    return ''

def extract_tumor_composition(text):
    """Extract tumor composition from report text."""
    text_lower = text.lower()
    
    has_necrosis = bool(re.search(r'\bnecros[ie]s\b|\bnecrotic\b', text_lower))
    has_cystic = bool(re.search(r'\bcystic\b', text_lower))
    has_solid = bool(re.search(r'\bsolid\b|\bhomogeneous enhancement\b|\bhomogenous enhancement\b', text_lower))
    
    if has_necrosis and has_cystic:
        return 'necrotic/ cystic tumor'
    elif has_necrosis:
        return 'necrotic tumor'
    elif has_cystic:
        return 'cystic tumor'
    elif has_solid:
        return 'none'  # solid enhancing, no necrosis/cyst
    
    return 'none'

def extract_enhancement_pattern(text):
    """Extract tumor enhancement pattern from report text."""
    text_lower = text.lower()
    
    features = []
    
    if re.search(r'\bheterogeneous\b.*\benhancement\b|\benhancement\b.*\bheterogeneous\b', text_lower):
        if re.search(r'\bperipheral\b', text_lower):
            if re.search(r'\birregular\b', text_lower):
                return 'heterogeneous peripheral irregular enhancement'
            if re.search(r'\bring\b', text_lower):
                return 'heterogeneous peripheral ring enhancement'
            if re.search(r'\bwall\b', text_lower):
                return 'heterogeneous peripheral wall enhancement'
            return 'heterogeneous peripheral enhancement'
        return 'heterogeneous enhancement'
    
    is_peripheral = bool(re.search(r'\bperipheral\b', text_lower))
    is_ring = bool(re.search(r'\bring\b', text_lower))
    is_irregular = bool(re.search(r'\birregular\b', text_lower))
    is_wall = bool(re.search(r'\bwall\b', text_lower))
    is_uniform = bool(re.search(r'\buniform\b|\brelatively smooth\b|\bsmooth\b', text_lower))
    is_nodular = bool(re.search(r'\bnodular\b', text_lower))
    
    if is_peripheral and is_irregular and is_ring and is_wall:
        return 'peripheral irregular wall ring enhancement'
    if is_peripheral and is_irregular and is_ring:
        return 'peripheral irregular ring enhancement'
    if is_peripheral and is_irregular and is_wall:
        return 'peripheral irregular wall enhancement'
    if is_peripheral and is_ring and is_nodular:
        return 'peripheral ring enhancement nodular enhancement'
    if is_peripheral and is_ring:
        return 'peripheral ring enhancement'
    if is_peripheral and is_irregular:
        return 'peripheral irregular enhancement'
    if is_peripheral and is_wall:
        return 'peripheral wall enhancement'
    if is_peripheral and is_uniform:
        return 'peripheral uniform wall enhancement'
    if is_peripheral:
        return 'peripheral enhancement'
    if is_ring:
        return 'ring enhancement'
    if is_irregular:
        return 'irregular enhancement'
    
    return 'heterogeneous enhancement'

def extract_surrounding_effects(text):
    """Extract surrounding effects (edema, mass effect, midline shift)."""
    text_lower = text.lower()
    
    # Edema level
    if re.search(r'\bmarked\s+(?:peritumoral|perilesional|perilesional|disproportionate)?\s*edema\b|\bmarked\s+edema\b|\bmarked\s+peri', text_lower):
        edema = 'marked edema'
    elif re.search(r'\bmoderate\s+(?:peritumoral|perilesional)?\s*edema\b', text_lower):
        edema = 'moderate edema'
    elif re.search(r'\bmild\s+(?:peritumoral|perilesional)?\s*edema\b|\bmild\s+peri', text_lower):
        edema = 'mild edema'
    elif re.search(r'\bno\s+(?:significant\s+)?(?:peritumoral|perilesional)?\s*edema\b', text_lower):
        edema = 'no edema'
    elif re.search(r'\bdisproportionate\b.*\bedema\b', text_lower):
        edema = 'marked edema'
    elif re.search(r'\bedema\b', text_lower):
        edema = 'mild edema'
    else:
        edema = 'no edema'
    
    # Mass effect
    if re.search(r'\bno\s+(?:significant\s+)?mass\s+effect\b', text_lower):
        mass = 'no mass effect'
    elif re.search(r'\b(?:marked|significant|adjacent)\s+mass\s+effect\b|\bcompression\b.*\bmidline\b', text_lower):
        mass = 'mass effect'
    elif re.search(r'\bmass\s+effect\b', text_lower):
        mass = 'mass effect'
    elif re.search(r'\bcompression\b', text_lower):
        mass = 'mass effect'
    else:
        mass = 'no mass effect'
    
    # Midline shift
    if re.search(r'\bcontralateral\s+midline\s+shift\b', text_lower):
        shift = 'with midline shift'
    elif re.search(r'\bmidline\s+shift\b', text_lower):
        if re.search(r'\bno\s+(?:significant\s+)?midline\s+shift\b', text_lower):
            shift = 'no midline shift'
        else:
            shift = 'with midline shift'
    else:
        shift = 'no midline shift'
    
    if shift == 'with midline shift':
        return f"{edema}, {mass} with midline shift"
    else:
        return f"{edema}, {mass}, no midline shift"

def extract_ventricular_involvement(text):
    """Extract ventricular/brainstem involvement."""
    text_lower = text.lower()
    
    # First check for explicit NO compression
    if re.search(r'\bno\s+(?:significant\s+)?compression\s+(?:on|of|in)\s+(?:the\s+)?ventricle', text_lower):
        # Only return no compression if there's no POSITIVE compression elsewhere
        if not re.search(r'(?:causing|with|and)\s+(?:mild\s+)?compression', text_lower):
            return 'no compression'
    
    involvements = []
    
    # Broad check: does text mention lateral ventricle AND compression/effacement anywhere?
    has_lat_vent = bool(re.search(r'\blateral\s+ventricle', text_lower))
    has_compression = bool(re.search(r'\bcompression\b|\bcompressing\b|\beffacement\b', text_lower))
    
    # Specific horn patterns (anterior horn, occipital horn, orbital horn, etc.)
    horn_match = re.search(r'\bcompression\s+(?:on|of|in)\s+(?:the\s+)?(?:\w+\s+)?horn\s+of\s+(?:both\s+)?(?:the\s+)?(?:right|left)?\s*lateral\s+ventricle', text_lower)
    if horn_match:
        if 'both' in horn_match.group():
            involvements.append('compression on lateral ventricles')
        else:
            involvements.append('compression on lateral ventricle')
    
    # Direct compression on lateral ventricle
    elif re.search(r'\bcompression\s+(?:on|of|in)\s+(?:the\s+)?(?:right|left)?\s*lateral\s+ventricle', text_lower):
        involvements.append('compression on lateral ventricle')
    elif re.search(r'\bcompression\s+(?:on|of|in)\s+(?:the\s+)?(?:both\s+)?lateral\s+ventricles', text_lower):
        involvements.append('compression on lateral ventricles')
    
    # Compression mentioned with ventricle in broader context
    # "compression of brain parenchyma and the lateral ventricle"
    # "compression ... as well as the right lateral ventricle"
    elif has_lat_vent and has_compression:
        if re.search(r'\bno\s+(?:significant\s+)?compression\b.*\bventricle\b', text_lower) and not re.search(r'compression\s+(?:on|of)', text_lower):
            pass  # Actually no compression
        else:
            involvements.append('compression on lateral ventricle')
    
    # General ventricle compression
    elif re.search(r'\bcompression\s+(?:on|of|in)\s+(?:the\s+)?ventricles?\b', text_lower):
        involvements.append('compression on ventricle')
    
    # Third ventricle
    if re.search(r'\bthird\s+ventricle\b', text_lower) and has_compression:
        if not re.search(r'\bno\b.*\bthird\s+ventricle\b', text_lower):
            involvements.append('compression on third ventricle')
    
    # Brainstem / midbrain
    if re.search(r'\b(?:brain\s*stem|brainstem|midbrain)\b', text_lower) and has_compression:
        if re.search(r'\bcompression\s+(?:on|of)\s+(?:the\s+)?(?:brain|mid)', text_lower) or re.search(r'\bcompressing\s+(?:the\s+)?midbrain', text_lower):
            involvements.append('compression on brainstem')
    
    # Corpus callosum
    if re.search(r'\bcorpus\s+callosum\b', text_lower):
        if re.search(r'\bcompression\s+(?:on|of)\s+(?:the\s+)?corpus\b|\binvolving\s+(?:the\s+)?corpus\b|\bsplenium\b|\bsphenium\b', text_lower):
            involvements.append('compression on corpus callosum')
    
    if involvements:
        return ', '.join(involvements)
    
    return 'no compression'

def extract_axis_shift(text):
    """Extract axis shift from report text."""
    text_lower = text.lower()
    
    if re.search(r'\bcontralateral\s+midline\s+shift\b', text_lower):
        return 'contralateral midline shift'
    if re.search(r'\bmidline\s+shift\b', text_lower):
        if re.search(r'\bno\s+(?:significant\s+)?midline\s+shift\b', text_lower):
            # Check if there's ALSO a positive midline shift mentioned elsewhere
            no_shift_spans = [(m.start(), m.end()) for m in re.finditer(r'\bno\s+(?:significant\s+)?midline\s+shift\b', text_lower)]
            all_shift_spans = [(m.start(), m.end()) for m in re.finditer(r'\bmidline\s+shift\b', text_lower)]
            # If there are midline shift mentions NOT covered by "no midline shift"
            positive_shifts = [s for s in all_shift_spans if not any(ns[0] <= s[0] and ns[1] >= s[1] for ns in no_shift_spans)]
            if positive_shifts:
                return 'contralateral midline shift'
            return 'no axis shift'
        return 'contralateral midline shift'
    return 'no axis shift'

def extract_diagnosis(text):
    """Extract histopathology/differential diagnosis."""
    text_lower = text.lower()
    
    diagnoses = []
    
    # GBM patterns
    if re.search(r'\bglioblastoma\s+multiforme\b|\bgbm\b', text_lower):
        diagnoses.append('glioblastoma multiforme (GBM)')
    
    # Glioma patterns
    if re.search(r'\bhigh[\s-]*grade\s+glioma\b', text_lower):
        if 'glioblastoma multiforme (GBM)' not in diagnoses:
            diagnoses.append('high-grade glioma')
    elif re.search(r'\blow[\s-]*grade\s+glioma\b', text_lower):
        diagnoses.append('low-grade glioma')
    elif re.search(r'\bglioma\b', text_lower):
        if 'glioblastoma multiforme (GBM)' not in diagnoses:
            diagnoses.append('glioma')
    
    # Astrocytoma
    if re.search(r'\bastrocytoma\b', text_lower):
        if re.search(r'\bcystic\s+astrocytoma\b', text_lower):
            diagnoses.append('cystic astrocytoma')
        elif re.search(r'\baggressive\s+astrocytoma\b', text_lower):
            diagnoses.append('aggressive astrocytoma')
        else:
            diagnoses.append('astrocytoma')
    
    # Metastasis
    if re.search(r'\bmetast(?:asis|atic|ases)\b', text_lower):
        if re.search(r'\bcystic\s+metast\b', text_lower):
            diagnoses.append('cystic metastasis')
        else:
            diagnoses.append('metastasis')
    
    # Meningioma
    if re.search(r'\bmeningioma\b', text_lower):
        diagnoses.append('meningioma')
    
    # Abscess
    if re.search(r'\babscess\b', text_lower):
        diagnoses.append('abscess')
    
    # SOL
    if re.search(r'\bsol\b|\bspace.occupying\b', text_lower):
        diagnoses.append('SOL')
    
    # Butterfly glioma
    if re.search(r'\bbutterfly\b', text_lower):
        diagnoses.append('butterfly glioma')
    
    # Tuberculoma
    if re.search(r'\btuberculoma\b', text_lower):
        diagnoses.append('tuberculoma')
    
    if diagnoses:
        return ' / '.join(diagnoses)
    return 'brain neoplasm'


###############################################################################
# 4. REPORT REWRITING
###############################################################################

def rewrite_report(corrected_transcript):
    """Convert corrected transcript into proper clinical report format.
    Follows the concise style of TR01-TR180 in the master file."""
    
    text = corrected_transcript.strip()
    
    # Remove conversational artifacts
    text = re.sub(r'^it\s+shows\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^there\s+is\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^it\s+is\s+', '', text, flags=re.IGNORECASE)
    
    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]
    
    # Ensure proper sentence endings
    if text and not text.endswith('.'):
        text += '.'
    
    # Clean up spacing
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+\.', '.', text)
    text = re.sub(r'\.{2,}', '.', text)
    
    return text.strip()


###############################################################################
# 5. MAIN PIPELINE
###############################################################################

def main():
    # Load transcripts
    df = pd.read_csv("/mnt/user-data/uploads/_transcripts.csv")
    
    # Load master file for existing reports
    doc = Document("/mnt/user-data/uploads/brain_tumor_classification_copy.numbers")
    sheet = doc.sheets[1]
    table = sheet.tables[0]
    
    existing_reports = {}
    for r in range(1, table.num_rows):
        num = str(table.cell(r, 0).value).strip()
        report = table.cell(r, 1).value
        if report and str(report).strip():
            existing_reports[num] = str(report).strip()
    
    # Process each transcript
    results = []
    duplicate_handling = {}
    
    for idx in range(len(df)):
        tr_num = CASE_MAP.get(idx)
        
        if not tr_num or tr_num == 'SKIP':
            continue
        
        # Handle duplicates - keep the longer transcript
        if tr_num.endswith('_DUP'):
            base_tr = tr_num.replace('_DUP', '')
            dup_text = str(df.iloc[idx]['transcript']) if pd.notna(df.iloc[idx]['transcript']) else ''
            if base_tr in duplicate_handling:
                if len(dup_text) > len(duplicate_handling[base_tr]):
                    duplicate_handling[base_tr] = dup_text
            else:
                duplicate_handling[base_tr] = dup_text
            continue
        
        raw_transcript = str(df.iloc[idx]['transcript']) if pd.notna(df.iloc[idx]['transcript']) else ''
        
        # Store for duplicate comparison
        if tr_num not in duplicate_handling:
            duplicate_handling[tr_num] = raw_transcript
        elif len(raw_transcript) > len(duplicate_handling[tr_num]):
            duplicate_handling[tr_num] = raw_transcript
    
    # Now process unique cases
    processed = {}
    for idx in range(len(df)):
        tr_num = CASE_MAP.get(idx)
        if not tr_num or tr_num == 'SKIP' or tr_num.endswith('_DUP'):
            continue
        if tr_num in processed:
            continue
        
        # Get the best transcript (longest if duplicates exist)
        raw_transcript = duplicate_handling.get(tr_num, str(df.iloc[idx]['transcript']))
        
        # Step 1: Correct ASR errors
        corrected = correct_asr_medical(raw_transcript)
        
        # Step 2: Determine final report
        # Use existing master file report if it exists and is non-empty
        if tr_num in existing_reports:
            final_report = existing_reports[tr_num]
        else:
            final_report = rewrite_report(corrected)
        
        # Step 3: Extract structured data
        # Use BOTH the existing report (cleaner) and corrected transcript (more detail)
        # Prefer existing report if available, use transcript for diagnosis info
        if tr_num in existing_reports:
            extraction_source = existing_reports[tr_num] + " " + corrected
        else:
            extraction_source = corrected if corrected else final_report
        
        hemisphere = extract_hemisphere(extraction_source)
        location = extract_tumor_location(extraction_source)
        composition = extract_tumor_composition(extraction_source)
        enhancement = extract_enhancement_pattern(extraction_source)
        effects = extract_surrounding_effects(extraction_source)
        ventricular = extract_ventricular_involvement(extraction_source)
        axis = extract_axis_shift(extraction_source)
        diagnosis = extract_diagnosis(extraction_source)
        
        results.append({
            'Number': tr_num,
            'Report': final_report,
            'Hemisphere': hemisphere,
            'Tumor Location': location,
            'Tumor Composition': composition,
            'Tumor Enhancement Pattern': enhancement,
            'Surrounding Effects': effects,
            'Ventricular/Brainstem Involvement': ventricular,
            'Axis Shift': axis,
            'Histopathology/Differential Diagnosis': diagnosis,
        })
        
        processed[tr_num] = True
    
    # Sort by TR number
    def sort_key(x):
        num = int(re.search(r'\d+', x['Number']).group())
        return num
    
    results.sort(key=sort_key)
    
    # Create DataFrame
    out_df = pd.DataFrame(results)
    
    # Save
    out_path = '/home/claude/neurofusion_transcripts_processed.csv'
    out_df.to_csv(out_path, index=False)
    
    print(f"Processed {len(results)} cases")
    print(f"Saved to {out_path}")
    print(f"\nCase number range: {results[0]['Number']} to {results[-1]['Number']}")
    print(f"\nSample output (TR184):")
    for r in results:
        if r['Number'] == 'TR184':
            for k, v in r.items():
                print(f"  {k}: {v}")
            break
    
    print(f"\nSample output (TR225 - new case):")
    for r in results:
        if r['Number'] == 'TR225':
            for k, v in r.items():
                print(f"  {k}: {v}")
            break
    
    print(f"\nSample output (TR301 - new batch):")
    for r in results:
        if r['Number'] == 'TR301':
            for k, v in r.items():
                print(f"  {k}: {v}")
            break

if __name__ == '__main__':
    main()
