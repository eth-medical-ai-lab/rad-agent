# Main used abnormality checklist
abnormality_dict_v4 = [
    "Check airways: in particular trachea (position, caliber, wall thickness), carina, main bronchi, bronchial thickening, bronchiectasis, bronchiolitis, mucoid impaction etc.",
    "Lung parenchyma assessment: check for nodules and masses, focal abnormalities, assess presence of diffuse patterns (ground-glass opacities, consolidation, reticular, nodular, etc)",
    "Pleural assessment: check for effusion (location, severity, associated findings), pneumothorax (approximate size, tension signs), pleural thickening (smooth vs. nodular, calcification, enhancement pattern)",
    "Heart: check pericardium (effusion, thickening, calcification), coronary arteries, cardiac chambers",
    "Cardiovascular & mediastinum: check aorta, atherosclerosis, pulmonary arteries (diameter of pulmonary trunk, patency if contrast-enhanced), and mediastinum (e.g. lymph nodes, thymus, esophagus, thyroid)",
    "Diaphragm & upper abdominal organs: diaphgram (position, defects, hernias), liver, adrenals, spleen, kidneys, pancreas, stomach. Note any abnormalities, focal lesions, masses, thickening etc.",
    "Spine, ribs, sternum, sternum & clavicles: check fractures, lesions, facet arthropathy, canal stenosis etc.",
    "Check chest wall, breasts, axillae, look for muscle asymmetry or masses, subcutaneous emphysema, nodules, edema etc.",
    "Check for presence of devices like catheters, tubes, lines, pacemakers, surgical clips etc. and note their position and any complications.",
]


# Detailed checklist, not used in the current version but can be used for more comprehensive analysis or as a reference for the model to ensure thoroughness in evaluation.
abnormality_dict = {
    "airways": {
        "trachea": " Position, caliber, wall thickness",
        "carina": "Angle (normal 60-80°), splaying suggests subcarinal mass",
        "main bronchi": "Patency, wall thickening, narrowing",
        "bronchiectasis": " Signet ring sign (bronchus > adjacent vessel)",
        "bronchiolitis": "tree-in-bud pattern, centrilobular nodules",
        "mucoid impaction": "Finger-in-glove appearance",
    },
    "breathing": {
        "lung parenchyma assessment": {
            "lung parenchyma focal abnormalities": {
                "nodules (<3cm)": "Size (measure in lung window), attenuation (solid, part-solid, pure ground-glass), morphology (smooth, lobulated, spiculated, corona radiata), internal characteristics (fat, calcification patterns), location (note if perifissural - likely benign)",
                "masses (≥3cm)": "Cavitation (wall thickness), air bronchograms, vessel involvement",
            },
            "lung parenchyma diffuse patterns": {
                "reticular": "Interlobular septal thickening, intralobular lines",
                "nodular": "Centrilobular, perilymphatic, random distribution",
                "ground-glass": "Pure vs. crazy-paving pattern",
                "consolidation": "Distribution (lobar, dependent, peribronchovascular)",
                "cystic/honeycomb": "Size, wall thickness, distribution (UIP pattern)",
                "mosaic attenuation": "Vascular vs. airway causes",
            },
        },
        "pleural assessment": {
            "effusion": "Quantify (small <10mm, moderate 10-30mm, large >30mm on lateral decubitus), density (simple, complex, hemothorax), associated findings (enhancing pleura, septations)",
            "pneumothorax": "Approximate size, tension signs (mediastinal shift, diaphragm inversion)",
            "thickening": "Smooth vs. nodular, calcification (plaques), enhancement pattern",
        },
    },
    "cardiovascular & mediastinum": {
        "heart": {
            "pericardium": "Effusion (measure in mm), thickening (>4mm abnormal), calcification",
            "coronary arteries": "Calcification (note if extensive), anomalies",
            "cardiac chambers": "Enlargement patterns",
        },
        "great vessels": {
            "aorta": "Systematic measurement (root <4cm, ascending <4cm, descending <3cm), dissection (intimal flap, true/false lumen), atherosclerosis (calcification, mural thrombus), variants (aberrant subclavian, coarctation)",
            "pulmonary arteries": "Main PA diameter (<3cm normal), ratio to aorta, check for filling defects (PE), chronic findings (webs, bands, mosaic perfusion)",
            "superior/inferior vena cava": "Patency, filling defects, collaterals",
        },
        "mediastinum": {
            "lymph nodes": "Measure short axis (stations 1–14 IASLC), abnormal if >1cm (>1.5cm subcarinal), internal characteristics (calcification, necrosis, fat)",
            "thymus": "Normal for age vs. hyperplasia vs. mass",
            "esophagus": "Wall thickness, dilation, fluid level",
            "thyroid": "If visible, note goiter or nodules",
        },
    },
    "diaphragm & upper abdomen": {
        "diaphragm": "Position (elevation, eventration), defects (Bochdalek, Morgagni, hiatal), motion (if inspiratory/expiratory phases available)",
        "upper abdominal organs": {
            "liver": "Size, focal lesions, steatosis",
            "adrenals": "Nodules (>1cm), calcification, fat content",
            "spleen": "Size, focal lesions",
            "kidneys": "Cysts, masses, hydronephrosis",
            "pancreas": "Focal lesions, ductal dilation (if visible)",
            "stomach": "Distention, wall thickening",
        },
    },
    "bones & soft tissues": {
        "osseous structures": {
            "spine": "Vertebral body height loss, lytic/sclerotic lesions, posterior element fractures, facet arthropathy, canal stenosis (if visible)",
            "ribs": "Count ribs, fractures (acute vs. chronic), lytic or sclerotic lesions",
            "sternum & clavicles": "Fractures, lesions, joints",
            "shoulder girdles": "Glenohumeral and AC joints",
        },
        "soft tissues": {
            "chest wall": "Muscle asymmetry or masses, subcutaneous emphysema, nodules, edema",
            "breasts": "Asymmetry, masses, calcifications (if visible)",
            "axillae": "Lymphadenopathy (level I–III)",
            "neck base": "Thyroid and lower cervical nodes",
        },
    },
}
