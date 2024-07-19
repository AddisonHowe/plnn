import numpy as np

DATDIRBASE = "data/training_data"

dir_facs_1a = f"{DATDIRBASE}/facs/pca/dec1/transition1_subset_epi_tr_ce_an_pc12"
dir_facs_1b = f"{DATDIRBASE}/facs/pca/dec1_fitonsubset/transition1_subset_epi_tr_ce_an_pc12"
dir_facs_2a = f"{DATDIRBASE}/facs/pca/dec2/transition2_subset_ce_pn_m_pc12"
dir_facs_2b = f"{DATDIRBASE}/facs/pca/dec2_fitonsubset/transition2_subset_ce_pn_m_pc12"

dir_facs_v2_1a = f"{DATDIRBASE}/facs_v2/pca/dec1/transition1_subset_epi_tr_ce_an_pc12"
dir_facs_v2_1b = f"{DATDIRBASE}/facs_v2/pca/dec1_fitonsubset/transition1_subset_epi_tr_ce_an_pc12"
dir_facs_v2_2a = f"{DATDIRBASE}/facs_v2/pca/dec2/transition2_subset_ce_pn_m_pc12"
dir_facs_v2_2b = f"{DATDIRBASE}/facs_v2/pca/dec2_fitonsubset/transition2_subset_ce_pn_m_pc12"

dir_facs_v3_1a = f"{DATDIRBASE}/facs_v3/pca/dec1/transition1_subset_epi_tr_ce_an_pc12"
dir_facs_v3_1b = f"{DATDIRBASE}/facs_v3/pca/dec1_fitonsubset/transition1_subset_epi_tr_ce_an_pc12"
dir_facs_v3_2a = f"{DATDIRBASE}/facs_v3/pca/dec2/transition2_subset_ce_pn_m_pc12"
dir_facs_v3_2b = f"{DATDIRBASE}/facs_v3/pca/dec2_fitonsubset/transition2_subset_ce_pn_m_pc12"

dir_facs_v4_2a = f"{DATDIRBASE}/facs_v4/pca/dec2/transition2_subset_ce_pn_m_pc12"
dir_facs_v4_2b = f"{DATDIRBASE}/facs_v4/pca/dec2_fitonsubset/transition2_subset_ce_pn_m_pc12"

SUBDIRMAP = {
    dir_facs_1a : ['training', 'validation'],
    dir_facs_1b : ['training', 'validation'],
    dir_facs_2a : ['training', 'validation'],
    dir_facs_2b : ['training', 'validation'],
    dir_facs_v2_1a : ['training', 'validation'],
    dir_facs_v2_1b : ['training', 'validation'],
    dir_facs_v2_2a : ['training', 'validation'],
    dir_facs_v2_2b : ['training', 'validation'],
    dir_facs_v3_1a : ['training', 'validation', 'testing'],
    dir_facs_v3_1b : ['training', 'validation', 'testing'],
    dir_facs_v3_2a : ['training', 'validation', 'testing'],
    dir_facs_v3_2b : ['training', 'validation', 'testing'],
    dir_facs_v4_2a : ['training', 'validation', 'testing'],
    dir_facs_v4_2b : ['training', 'validation', 'testing'],
}

EXPECTED_CONDITIONS_MAP = {
    (dir_facs_1a, 'training') : [
        [[0, 0, 0, 10],[1, 1, 0.9,  10]],  # NO CHIR
        [[1, 1, 0, 10],[1, 1, 0.9,  10]],  # CHIR 2-3
        [[2, 1, 0, 10],[1, 1, 0.9,  10]],  # CHIR 2-4
        [[3, 1, 0, 10],[1, 1, 0.9,  10]],  # CHIR 2-5
        [[3, 1, 0, 10],[1, 1, 0,    10]],  # CHIR 2-5 FGF 2-3
        [[3, 1, 0, 10],[2, 1, 0,    10]],  # CHIR 2-5 FGF 2-4
        [[3, 1, 0, 10],[3, 1, 0.9,  10]],  # CHIR 2-5 FGF 2-5
    ],
    (dir_facs_1a, 'validation') : [
        [[0.5, 1, 0, 10],[1, 1, 0.9,  10]],  # CHIR 2-2.5
        [[1.5, 1, 0, 10],[1, 1, 0.9,  10]],  # CHIR 2-3.5
        [[3, 1, 0, 10],[1.5, 1, 0,    10]],  # CHIR 2-5 FGF 2-3.5
        [[3, 1, 0, 10],[2.5, 1, 0,    10]],  # CHIR 2-5 FGF 2-4.5
    ],
    (dir_facs_1b, 'training') : [
        [[0, 0, 0, 10],[1, 1, 0.9,  10]],  # NO CHIR
        [[1, 1, 0, 10],[1, 1, 0.9,  10]],  # CHIR 2-3
        [[2, 1, 0, 10],[1, 1, 0.9,  10]],  # CHIR 2-4
        [[3, 1, 0, 10],[1, 1, 0.9,  10]],  # CHIR 2-5
        [[3, 1, 0, 10],[1, 1, 0,    10]],  # CHIR 2-5 FGF 2-3
        [[3, 1, 0, 10],[2, 1, 0,    10]],  # CHIR 2-5 FGF 2-4
        [[3, 1, 0, 10],[3, 1, 0.9,  10]],  # CHIR 2-5 FGF 2-5
    ],
    (dir_facs_1b, 'validation') : [
        [[0.5, 1, 0, 10],[1, 1, 0.9,  10]],  # CHIR 2-2.5
        [[1.5, 1, 0, 10],[1, 1, 0.9,  10]],  # CHIR 2-3.5
        [[3, 1, 0, 10],[1.5, 1, 0,    10]],  # CHIR 2-5 FGF 2-3.5
        [[3, 1, 0, 10],[2.5, 1, 0,    10]],  # CHIR 2-5 FGF 2-4.5
    ],



    (dir_facs_2a, 'training') : [
        # [[0, 0, 0, 10],[1, 1, 0.9,  10]],  # REMOVE NO CHIR
        [[1, 1, 0, 10],[1, 1, 0.9,  10]],  # CHIR 2-3
        [[2, 1, 0, 10],[1, 1, 0.9,  10]],  # CHIR 2-4
        [[3, 1, 0, 10],[1, 1, 0.9,  10]],  # CHIR 2-5
        [[3, 1, 0, 10],[1, 1, 0,    10]],  # CHIR 2-5 FGF 2-3
        [[3, 1, 0, 10],[2, 1, 0,    10]],  # CHIR 2-5 FGF 2-4
        [[3, 1, 0, 10],[3, 1, 0.9,  10]],  # CHIR 2-5 FGF 2-5
    ],
    (dir_facs_2a, 'validation') : [
        [[0.5, 1, 0, 10],[1, 1, 0.9,  10]],  # CHIR 2-2.5
        [[1.5, 1, 0, 10],[1, 1, 0.9,  10]],  # CHIR 2-3.5
        [[3, 1, 0, 10],[1.5, 1, 0,    10]],  # CHIR 2-5 FGF 2-3.5
        [[3, 1, 0, 10],[2.5, 1, 0,    10]],  # CHIR 2-5 FGF 2-4.5
    ],
    (dir_facs_2b, 'training') : [
        # [[0, 0, 0, 10],[1, 1, 0.9,  10]],  # REMOVE NO CHIR
        [[1, 1, 0, 10],[1, 1, 0.9,  10]],  # CHIR 2-3
        [[2, 1, 0, 10],[1, 1, 0.9,  10]],  # CHIR 2-4
        [[3, 1, 0, 10],[1, 1, 0.9,  10]],  # CHIR 2-5
        [[3, 1, 0, 10],[1, 1, 0,    10]],  # CHIR 2-5 FGF 2-3
        [[3, 1, 0, 10],[2, 1, 0,    10]],  # CHIR 2-5 FGF 2-4
        [[3, 1, 0, 10],[3, 1, 0.9,  10]],  # CHIR 2-5 FGF 2-5
    ],
    (dir_facs_2b, 'validation') : [
        [[0.5, 1, 0, 10],[1, 1, 0.9,  10]],  # CHIR 2-2.5
        [[1.5, 1, 0, 10],[1, 1, 0.9,  10]],  # CHIR 2-3.5
        [[3, 1, 0, 10],[1.5, 1, 0,    10]],  # CHIR 2-5 FGF 2-3.5
        [[3, 1, 0, 10],[2.5, 1, 0,    10]],  # CHIR 2-5 FGF 2-4.5
    ],


    (dir_facs_v2_1a, 'training') : [
        [[0, 0, 0, 1000],[1, 1, 0.9,  1000]],  # NO CHIR
        [[1, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-3
        [[2, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-4
        [[3, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-5
        [[3, 1, 0, 1000],[1, 1, 0,    1000]],  # CHIR 2-5 FGF 2-3
        [[3, 1, 0, 1000],[2, 1, 0,    1000]],  # CHIR 2-5 FGF 2-4
        [[3, 1, 0, 1000],[3, 1, 0.9,  1000]],  # CHIR 2-5 FGF 2-5
    ],
    (dir_facs_v2_1a, 'validation') : [
        [[0.5, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-2.5
        [[1.5, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-3.5
        [[3, 1, 0, 1000],[1.5, 1, 0,    1000]],  # CHIR 2-5 FGF 2-3.5
        [[3, 1, 0, 1000],[2.5, 1, 0,    1000]],  # CHIR 2-5 FGF 2-4.5
    ],
    (dir_facs_v2_1b, 'training') : [
        [[0, 0, 0, 1000],[1, 1, 0.9,  1000]],  # NO CHIR
        [[1, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-3
        [[2, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-4
        [[3, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-5
        [[3, 1, 0, 1000],[1, 1, 0,    1000]],  # CHIR 2-5 FGF 2-3
        [[3, 1, 0, 1000],[2, 1, 0,    1000]],  # CHIR 2-5 FGF 2-4
        [[3, 1, 0, 1000],[3, 1, 0.9,  1000]],  # CHIR 2-5 FGF 2-5
    ],
    (dir_facs_v2_1b, 'validation') : [
        [[0.5, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-2.5
        [[1.5, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-3.5
        [[3, 1, 0, 1000],[1.5, 1, 0,    1000]],  # CHIR 2-5 FGF 2-3.5
        [[3, 1, 0, 1000],[2.5, 1, 0,    1000]],  # CHIR 2-5 FGF 2-4.5
    ],
    (dir_facs_v2_2a, 'training') : [
        # [[0, 0, 0, 1000],[1, 1, 0.9,  1000]],  # REMOVE NO CHIR
        [[1, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-3
        [[2, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-4
        [[3, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-5
        [[3, 1, 0, 1000],[1, 1, 0,    1000]],  # CHIR 2-5 FGF 2-3
        [[3, 1, 0, 1000],[2, 1, 0,    1000]],  # CHIR 2-5 FGF 2-4
        [[3, 1, 0, 1000],[3, 1, 0.9,  1000]],  # CHIR 2-5 FGF 2-5
    ],
    (dir_facs_v2_2a, 'validation') : [
        [[0.5, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-2.5
        [[1.5, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-3.5
        [[3, 1, 0, 1000],[1.5, 1, 0,    1000]],  # CHIR 2-5 FGF 2-3.5
        [[3, 1, 0, 1000],[2.5, 1, 0,    1000]],  # CHIR 2-5 FGF 2-4.5
    ],
    (dir_facs_v2_2b, 'training') : [
        # [[0, 0, 0, 1000],[1, 1, 0.9,  1000]],  # REMOVE NO CHIR
        [[1, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-3
        [[2, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-4
        [[3, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-5
        [[3, 1, 0, 1000],[1, 1, 0,    1000]],  # CHIR 2-5 FGF 2-3
        [[3, 1, 0, 1000],[2, 1, 0,    1000]],  # CHIR 2-5 FGF 2-4
        [[3, 1, 0, 1000],[3, 1, 0.9,  1000]],  # CHIR 2-5 FGF 2-5
    ],
    (dir_facs_v2_2b, 'validation') : [
        [[0.5, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-2.5
        [[1.5, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-3.5
        [[3, 1, 0, 1000],[1.5, 1, 0,    1000]],  # CHIR 2-5 FGF 2-3.5
        [[3, 1, 0, 1000],[2.5, 1, 0,    1000]],  # CHIR 2-5 FGF 2-4.5
    ],


    (dir_facs_v3_1a, 'training') : [
        [[0, 0, 0, 1000],[1, 1, 0.9,  1000]],  # NO CHIR
        [[1, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-3
        [[3, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-5
        [[3, 1, 0, 1000],[1, 1, 0,    1000]],  # CHIR 2-5 FGF 2-3
        [[3, 1, 0, 1000],[3, 1, 0.9,  1000]],  # CHIR 2-5 FGF 2-5
    ],
    (dir_facs_v3_1a, 'validation') : [
        [[0.5, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-2.5
        [[1.5, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-3.5
        [[3, 1, 0, 1000],[1.5, 1, 0,    1000]],  # CHIR 2-5 FGF 2-3.5
        [[3, 1, 0, 1000],[2.5, 1, 0,    1000]],  # CHIR 2-5 FGF 2-4.5
    ],
    (dir_facs_v3_1a, 'testing') : [
        [[2, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-4
        [[3, 1, 0, 1000],[2, 1, 0,    1000]],  # CHIR 2-5 FGF 2-4
    ],
    (dir_facs_v3_1b, 'training') : [
        [[0, 0, 0, 1000],[1, 1, 0.9,  1000]],  # NO CHIR
        [[1, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-3
        [[3, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-5
        [[3, 1, 0, 1000],[1, 1, 0,    1000]],  # CHIR 2-5 FGF 2-3
        [[3, 1, 0, 1000],[3, 1, 0.9,  1000]],  # CHIR 2-5 FGF 2-5
    ],
    (dir_facs_v3_1b, 'validation') : [
        [[0.5, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-2.5
        [[1.5, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-3.5
        [[3, 1, 0, 1000],[1.5, 1, 0,    1000]],  # CHIR 2-5 FGF 2-3.5
        [[3, 1, 0, 1000],[2.5, 1, 0,    1000]],  # CHIR 2-5 FGF 2-4.5
    ],
    (dir_facs_v3_1b, 'testing') : [
        [[2, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-4
        [[3, 1, 0, 1000],[2, 1, 0,    1000]],  # CHIR 2-5 FGF 2-4
    ],
    (dir_facs_v3_2a, 'training') : [
        # [[0, 0, 0, 1000],[1, 1, 0.9,  1000]],  # REMOVE NO CHIR
        [[1, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-3
        [[3, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-5
        [[3, 1, 0, 1000],[1, 1, 0,    1000]],  # CHIR 2-5 FGF 2-3
        [[3, 1, 0, 1000],[3, 1, 0.9,  1000]],  # CHIR 2-5 FGF 2-5
    ],
    (dir_facs_v3_2a, 'validation') : [
        [[0.5, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-2.5
        [[1.5, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-3.5
        [[3, 1, 0, 1000],[1.5, 1, 0,    1000]],  # CHIR 2-5 FGF 2-3.5
        [[3, 1, 0, 1000],[2.5, 1, 0,    1000]],  # CHIR 2-5 FGF 2-4.5
    ],
    (dir_facs_v3_2a, 'testing') : [
        [[2, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-4
        [[3, 1, 0, 1000],[2, 1, 0,    1000]],  # CHIR 2-5 FGF 2-4
    ],
    (dir_facs_v3_2b, 'training') : [
        # [[0, 0, 0, 1000],[1, 1, 0.9,  1000]],  # REMOVE NO CHIR
        [[1, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-3
        [[3, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-5
        [[3, 1, 0, 1000],[1, 1, 0,    1000]],  # CHIR 2-5 FGF 2-3
        [[3, 1, 0, 1000],[3, 1, 0.9,  1000]],  # CHIR 2-5 FGF 2-5
    ],
    (dir_facs_v3_2b, 'validation') : [
        [[0.5, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-2.5
        [[1.5, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-3.5
        [[3, 1, 0, 1000],[1.5, 1, 0,    1000]],  # CHIR 2-5 FGF 2-3.5
        [[3, 1, 0, 1000],[2.5, 1, 0,    1000]],  # CHIR 2-5 FGF 2-4.5
    ],
    (dir_facs_v3_2b, 'testing') : [
        [[2, 1, 0, 1000],[1, 1, 0.9,  1000]],  # CHIR 2-4
        [[3, 1, 0, 1000],[2, 1, 0,    1000]],  # CHIR 2-5 FGF 2-4
    ],


    (dir_facs_v4_2a, 'training') : [
        # [[0, 0, 0, 1000],[1, 1, 0.9,  1000]],  # REMOVE NO CHIR
        [[0, 1, 0, 1000],[0, 1, 0.9,  1000]],  # CHIR 2-3
        [[2, 1, 0, 1000],[0, 1, 0.9,  1000]],  # CHIR 2-5
        [[2, 1, 0, 1000],[0, 1, 0,    1000]],  # CHIR 2-5 FGF 2-3
        [[2, 1, 0, 1000],[2, 1, 0.9,  1000]],  # CHIR 2-5 FGF 2-5
    ],
    (dir_facs_v4_2a, 'validation') : [
        [[-0.5, 1, 0, 1000],[0, 1, 0.9,  1000]],  # CHIR 2-2.5
        [[0.5, 1, 0, 1000],[0, 1, 0.9,  1000]],  # CHIR 2-3.5
        [[2, 1, 0, 1000],[0.5, 1, 0,    1000]],  # CHIR 2-5 FGF 2-3.5
        [[2, 1, 0, 1000],[1.5, 1, 0,    1000]],  # CHIR 2-5 FGF 2-4.5
    ],
    (dir_facs_v4_2a, 'testing') : [
        [[1, 1, 0, 1000],[0, 1, 0.9,  1000]],  # CHIR 2-4
        [[2, 1, 0, 1000],[1, 1, 0,    1000]],  # CHIR 2-5 FGF 2-4
    ],
    (dir_facs_v4_2b, 'training') : [
        # [[0, 0, 0, 1000],[0, 1, 0.9,  1000]],  # REMOVE NO CHIR
        [[0, 1, 0, 1000],[0, 1, 0.9,  1000]],  # CHIR 2-3
        [[2, 1, 0, 1000],[0, 1, 0.9,  1000]],  # CHIR 2-5
        [[2, 1, 0, 1000],[0, 1, 0,    1000]],  # CHIR 2-5 FGF 2-3
        [[2, 1, 0, 1000],[2, 1, 0.9,  1000]],  # CHIR 2-5 FGF 2-5
    ],
    (dir_facs_v4_2b, 'validation') : [
        [[-0.5, 1, 0, 1000],[0, 1, 0.9,  1000]],  # CHIR 2-2.5
        [[0.5, 1, 0, 1000],[0, 1, 0.9,  1000]],  # CHIR 2-3.5
        [[2, 1, 0, 1000],[0.5, 1, 0,    1000]],  # CHIR 2-5 FGF 2-3.5
        [[2, 1, 0, 1000],[1.5, 1, 0,    1000]],  # CHIR 2-5 FGF 2-4.5
    ],
    (dir_facs_v4_2b, 'testing') : [
        [[1, 1, 0, 1000],[0, 1, 0.9,  1000]],  # CHIR 2-4
        [[2, 1, 0, 1000],[1, 1, 0,    1000]],  # CHIR 2-5 FGF 2-4
    ],
}



errors = []
for datdir, subdir in EXPECTED_CONDITIONS_MAP:
    d = datdir + '/' + subdir
    sigparam_list = EXPECTED_CONDITIONS_MAP[(datdir, subdir)]
    nsims = len(sigparam_list)

    
    for simidx in range(nsims):
        sps = np.load(f"{d}/sim{simidx}/sigparams.npy")
        if not (np.allclose(sps, sigparam_list[simidx])):
            msg = f"Error in {d} : sim{simidx}.\n"
            msg += f"Expected:\n {sigparam_list[simidx]}\n"
            msg += f"Got:\n {sps}\n"
            errors.append(msg)



assert not errors, "Errors occurred:\n{}".format("\n".join(errors))