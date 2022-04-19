/*
From: 
 J. Chem. Inf. Model. 2009, 49, 84–96
Empirical Scoring Functions for Advanced Protein-Ligand Docking with PLANTS
Oliver Korb,† Thomas Stützle,‡ and Thomas E. Exner*,†
Theoretische Chemische Dynamik, Fachbereich Chemie, Universität Konstanz, 78457 Konstanz, Germany,
and IRIDIA, CoDE, Université Libre de Bruxelles, Brussels, Belgium

Using PLANTS Model5
*/

#ifndef PLP_CONSTANTS_CL_H
#define PLP_CONSTANTS_CL_H

enum CHROM_m_mode_eMode {
    CHROM_m_mode_eMode_FIXED = 0,
    CHROM_m_mode_eMode_TETHERED = 1,
    CHROM_m_mode_eMode_FREE = 2
};
typedef enum CHROM_m_mode_eMode CHROM_m_mode_eMode;

enum TRIPOS_TYPE {
    TRIPOS_TYPE_UNDEFINED = 0,
    TRIPOS_TYPE_Al,
    TRIPOS_TYPE_Br,
    TRIPOS_TYPE_C_cat,
    TRIPOS_TYPE_C_1,
    TRIPOS_TYPE_C_1_H1,
    TRIPOS_TYPE_C_2,
    TRIPOS_TYPE_C_2_H1,
    TRIPOS_TYPE_C_2_H2,
    TRIPOS_TYPE_C_3,
    TRIPOS_TYPE_C_3_H1,
    TRIPOS_TYPE_C_3_H2,
    TRIPOS_TYPE_C_3_H3,
    TRIPOS_TYPE_C_ar,
    TRIPOS_TYPE_C_ar_H1,
    TRIPOS_TYPE_Ca,
    TRIPOS_TYPE_Cl,
    TRIPOS_TYPE_Du,
    TRIPOS_TYPE_F,
    TRIPOS_TYPE_H,
    TRIPOS_TYPE_H_P,
    TRIPOS_TYPE_I,
    TRIPOS_TYPE_K,
    TRIPOS_TYPE_Li,
    TRIPOS_TYPE_LP,
    TRIPOS_TYPE_N_1,
    TRIPOS_TYPE_N_2,
    TRIPOS_TYPE_N_3,
    TRIPOS_TYPE_N_4,
    TRIPOS_TYPE_N_am,
    TRIPOS_TYPE_N_ar,
    TRIPOS_TYPE_N_pl3,
    TRIPOS_TYPE_Na,
    TRIPOS_TYPE_O_2,
    TRIPOS_TYPE_O_3,
    TRIPOS_TYPE_O_co2,
    TRIPOS_TYPE_P_3,
    TRIPOS_TYPE_S_2,
    TRIPOS_TYPE_S_3,
    TRIPOS_TYPE_S_o,
    TRIPOS_TYPE_S_o2,
    TRIPOS_TYPE_Si,
    TRIPOS_TYPE_MAXTYPES // KEEP AS LAST TYPE: used to size the atom name string list
};
typedef enum TRIPOS_TYPE TRIPOS_TYPE;

// Classification Types:
enum PLP_CLASS {
    PLP_CLASS_hdo = 0,
    PLP_CLASS_hac = 1,
    PLP_CLASS_hda = 2,
    PLP_CLASS_oil = 3,
    PLP_CLASS_met = 4,
    PLP_CLASS_ath = 5,
    PLP_CLASS_NO_CLASS = 6
};
typedef enum PLP_CLASS PLP_CLASS;

// Interaction Types
enum PLP_INTERACTION {
    PLP_INTERACTION_hbond = 0,
    PLP_INTERACTION_repulsive = 1,
    PLP_INTERACTION_buried = 2,
    PLP_INTERACTION_metal = 3,
    PLP_INTERACTION_nonpolar = 4,
    PLP_INTERACTION_NO_INTERACTION = 5
};
typedef enum PLP_INTERACTION PLP_INTERACTION;

// Clash Classification Types:
enum PLP_CLASH_CLASS {
    PLP_CLASH_CLASS_1 = 0,
    PLP_CLASH_CLASS_2 = 1,
    PLP_CLASH_CLASS_3 = 2
};
typedef enum PLP_CLASH_CLASS PLP_CLASH_CLASS;

//Model5 weights:
#define W_PLP_HB_M5 -2.00f
#define W_PLP_MET_M5 -4.00f
#define W_PLP_BUR_M5 -0.05f
#define W_PLP_NONP_M5 -0.40f
#define W_PLP_REP_M5 0.50f
// SINCE = 1.0, NOT TAKEN INTO ACCOUNT, IF CHANGED, FIX PROGRAM:
// W_PLP_TORS_M5 * sum(all_tors)    (now tors parts all put into "score" with all other parts).
#define W_PLP_TORS_M5 1.00f

// TABLES:

__constant int CLASSIFICATION_TABLE[43] = {
    PLP_CLASS_NO_CLASS,//TRIPOS_TYPE_UNDEFINED 0
    PLP_CLASS_met,//TRIPOS_TYPE_Al           1
    PLP_CLASS_oil,//TRIPOS_TYPE_Br           2
    PLP_CLASS_hdo,//TRIPOS_TYPE_C_cat        3
    PLP_CLASS_oil,//TRIPOS_TYPE_C_1          4
    PLP_CLASS_oil,//TRIPOS_TYPE_C_1_H1       5
    PLP_CLASS_oil,//TRIPOS_TYPE_C_2          6
    PLP_CLASS_oil,//TRIPOS_TYPE_C_2_H1       7
    PLP_CLASS_oil,//TRIPOS_TYPE_C_2_H2       8
    PLP_CLASS_oil,//TRIPOS_TYPE_C_3          9
    PLP_CLASS_oil,//TRIPOS_TYPE_C_3_H1       10
    PLP_CLASS_oil,//TRIPOS_TYPE_C_3_H2       11
    PLP_CLASS_oil,//TRIPOS_TYPE_C_3_H3       12
    PLP_CLASS_oil,//TRIPOS_TYPE_C_ar         13
    PLP_CLASS_oil,//TRIPOS_TYPE_C_ar_H1      14
    PLP_CLASS_met,//TRIPOS_TYPE_Ca           15
    PLP_CLASS_oil,//TRIPOS_TYPE_Cl           16
    PLP_CLASS_NO_CLASS,//TRIPOS_TYPE_Du      17
    PLP_CLASS_hac,//TRIPOS_TYPE_F            18
    PLP_CLASS_oil,//TRIPOS_TYPE_H            19
    PLP_CLASS_ath,//TRIPOS_TYPE_H_P          20
    PLP_CLASS_oil,//TRIPOS_TYPE_I            21
    PLP_CLASS_met,//TRIPOS_TYPE_K            22
    PLP_CLASS_met,//TRIPOS_TYPE_Li           23
    PLP_CLASS_hac,//TRIPOS_TYPE_LP           24
    PLP_CLASS_hac,//TRIPOS_TYPE_N_1          25
    PLP_CLASS_hda,//TRIPOS_TYPE_N_2          26
    PLP_CLASS_hda,//TRIPOS_TYPE_N_3          27
    PLP_CLASS_hdo,//TRIPOS_TYPE_N_4          28
    PLP_CLASS_hdo,//TRIPOS_TYPE_N_am         29
    PLP_CLASS_hda,//TRIPOS_TYPE_N_ar         30
    PLP_CLASS_hda,//TRIPOS_TYPE_N_pl3        31
    PLP_CLASS_met,//TRIPOS_TYPE_Na           32
    PLP_CLASS_hac,//TRIPOS_TYPE_O_2          33
    PLP_CLASS_hda,//TRIPOS_TYPE_O_3          34
    PLP_CLASS_hac,//TRIPOS_TYPE_O_co2        35
    PLP_CLASS_hda,//TRIPOS_TYPE_P_3          36
    PLP_CLASS_hda,//TRIPOS_TYPE_S_2          37
    PLP_CLASS_hda,//TRIPOS_TYPE_S_3          38
    PLP_CLASS_hac,//TRIPOS_TYPE_S_o          39
    PLP_CLASS_hac,//TRIPOS_TYPE_S_o2         40
    PLP_CLASS_met,//TRIPOS_TYPE_Si           41
    PLP_CLASS_NO_CLASS//TRIPOS_TYPE_MAXTYPES 42
};

__constant int INTERACTION_TABLE[49] = {
    PLP_INTERACTION_repulsive,// [0,0] [hdo,hdo] 0
    PLP_INTERACTION_hbond,// [0,1] [hdo,hac] 1
    PLP_INTERACTION_hbond,// [0,2] [hdo,hda] 2
    PLP_INTERACTION_buried,// [0,3] [hdo,oil] 3
    PLP_INTERACTION_repulsive,// [0,4] [hdo,met] 4
    PLP_INTERACTION_NO_INTERACTION,// [0,5] [hdo,ath] 5
    PLP_INTERACTION_NO_INTERACTION,// [0,6] [hdo,NO_I] 6
    PLP_INTERACTION_hbond,// [1,0] [hac,hdo] 7
    PLP_INTERACTION_repulsive,// [1,1] [hac,hac] 8
    PLP_INTERACTION_hbond,// [1,2] [hac,hda] 9
    PLP_INTERACTION_buried,// [1,3] [hac,oil] 10
    PLP_INTERACTION_metal,// [1,4] [hac,met] 11
    PLP_INTERACTION_NO_INTERACTION,// [1,5] [hac,ath] 12
    PLP_INTERACTION_NO_INTERACTION,// [1,6] [hac,NO_I] 13
    PLP_INTERACTION_hbond,// [2,0] [hda,hdo] 14
    PLP_INTERACTION_hbond,// [2,1] [hda,hac] 15
    PLP_INTERACTION_hbond,// [2,2] [hda,hda] 16
    PLP_INTERACTION_buried,// [2,3] [hda,oil] 17
    PLP_INTERACTION_metal,// [2,4] [hda,met] 18
    PLP_INTERACTION_NO_INTERACTION,// [2,5] [hda,ath] 19
    PLP_INTERACTION_NO_INTERACTION,// [2,6] [hda,NO_I] 20
    PLP_INTERACTION_buried,// [3,0] [oil,hdo] 21
    PLP_INTERACTION_buried,// [3,1] [oil,hac] 22
    PLP_INTERACTION_buried,// [3,2] [oil,hda] 23
    PLP_INTERACTION_nonpolar,// [3,3] [oil,oil] 24
    PLP_INTERACTION_buried,// [3,4] [oil,met] 25
    PLP_INTERACTION_NO_INTERACTION,// [3,5] [oil,ath] 26
    PLP_INTERACTION_NO_INTERACTION,// [3,6] [oil,NO_I] 27
    PLP_INTERACTION_repulsive,// [4,0] [met,hdo] 28
    PLP_INTERACTION_metal,// [4,1] [met,hac] 29
    PLP_INTERACTION_metal,// [4,2] [met,hda] 30
    PLP_INTERACTION_buried,// [4,3] [met,oil] 31
    PLP_INTERACTION_NO_INTERACTION,// [4,4] [met,met] 32
    PLP_INTERACTION_NO_INTERACTION,// [4,5] [met,ath] 33
    PLP_INTERACTION_NO_INTERACTION,// [4,6] [met,NO_I] 34
    PLP_INTERACTION_NO_INTERACTION,// [5,0] [ath,hdo] 35
    PLP_INTERACTION_NO_INTERACTION,// [5,1] [ath,hac] 36
    PLP_INTERACTION_NO_INTERACTION,// [5,2] [ath,hda] 37
    PLP_INTERACTION_NO_INTERACTION,// [5,3] [ath,oil] 38
    PLP_INTERACTION_NO_INTERACTION,// [5,4] [ath,met] 39
    PLP_INTERACTION_NO_INTERACTION,// [5,5] [ath,ath] 40
    PLP_INTERACTION_NO_INTERACTION,// [5,6] [ath,NO_I] 41
    PLP_INTERACTION_NO_INTERACTION,// [6,0] [NO_I,hdo] 42
    PLP_INTERACTION_NO_INTERACTION,// [6,1] [NO_I,hac] 43
    PLP_INTERACTION_NO_INTERACTION,// [6,2] [NO_I,hda] 44
    PLP_INTERACTION_NO_INTERACTION,// [6,3] [NO_I,oil] 45
    PLP_INTERACTION_NO_INTERACTION,// [6,4] [NO_I,met] 46
    PLP_INTERACTION_NO_INTERACTION,// [6,5] [NO_I,ath] 47
    PLP_INTERACTION_NO_INTERACTION// [6,6] [NO_I,NO_I] 48
};

__constant float FPLP_TABLE[36] = {
    2.3f,// A PLP_INTERACTION_hbond
    2.6f,// B PLP_INTERACTION_hbond
    3.1f,// C PLP_INTERACTION_hbond
    3.4f,// D PLP_INTERACTION_hbond
    W_PLP_HB_M5,// E PLP_INTERACTION_hbond
    20.0f,// F PLP_INTERACTION_hbond
    3.2f,// A PLP_INTERACTION_repulsive
    5.0f,// B PLP_INTERACTION_repulsive
    W_PLP_REP_M5 * 0.1f,// C PLP_INTERACTION_repulsive
    W_PLP_REP_M5 * 20.0f,// D PLP_INTERACTION_repulsive
    222.222f,// E PLP_INTERACTION_repulsive (Must be different than x to prevent devide by zero)
    333.333f,// F PLP_INTERACTION_repulsive (Must be different than x to prevent devide by zero)
    3.4f,// A PLP_INTERACTION_buried
    3.6f,// B PLP_INTERACTION_buried
    4.5f,// C PLP_INTERACTION_buried
    5.5f,// D PLP_INTERACTION_buried
    W_PLP_BUR_M5,// E PLP_INTERACTION_buried
    20.0f,// F PLP_INTERACTION_buried
    1.4f,// A PLP_INTERACTION_metal
    2.2f,// B PLP_INTERACTION_metal
    2.6f,// C PLP_INTERACTION_metal
    2.8f,// D PLP_INTERACTION_metal
    W_PLP_MET_M5,// E PLP_INTERACTION_metal
    20.0f,// F PLP_INTERACTION_metal
    3.4f,// A PLP_INTERACTION_nonpolar
    3.6f,// B PLP_INTERACTION_nonpolar
    4.5f,// C PLP_INTERACTION_nonpolar
    5.5f,// D PLP_INTERACTION_nonpolar
    W_PLP_NONP_M5,// E PLP_INTERACTION_nonpolar
    20.0f,// F PLP_INTERACTION_nonpolar
    999.999f,// A PLP_INTERACTION_NO_INTERACTION (Must be different than x to prevent devide by zero)
    888.888f,// B PLP_INTERACTION_NO_INTERACTION (Must be different than x to prevent devide by zero)
    777.777f,// C PLP_INTERACTION_NO_INTERACTION (Must be different than x to prevent devide by zero)
    666.666f,// D PLP_INTERACTION_NO_INTERACTION (Must be different than x to prevent devide by zero)
    555.555f,// E PLP_INTERACTION_NO_INTERACTION (Must be different than x to prevent devide by zero)
    444.444f// F PLP_INTERACTION_NO_INTERACTION (Must be different than x to prevent devide by zero)
};

__constant int CLASH_CLASSIFICATION_TABLE[43] = {
    PLP_CLASH_CLASS_3,//TRIPOS_TYPE_UNDEFINED 0
    PLP_CLASH_CLASS_3,//TRIPOS_TYPE_Al           1
    PLP_CLASH_CLASS_1,//TRIPOS_TYPE_Br           2
    PLP_CLASH_CLASS_3,//TRIPOS_TYPE_C_cat        3
    PLP_CLASH_CLASS_3,//TRIPOS_TYPE_C_1          4
    PLP_CLASH_CLASS_3,//TRIPOS_TYPE_C_1_H1       5
    PLP_CLASH_CLASS_3,//TRIPOS_TYPE_C_2          6
    PLP_CLASH_CLASS_3,//TRIPOS_TYPE_C_2_H1       7
    PLP_CLASH_CLASS_3,//TRIPOS_TYPE_C_2_H2       8
    PLP_CLASH_CLASS_3,//TRIPOS_TYPE_C_3          9
    PLP_CLASH_CLASS_3,//TRIPOS_TYPE_C_3_H1       10
    PLP_CLASH_CLASS_3,//TRIPOS_TYPE_C_3_H2       11
    PLP_CLASH_CLASS_3,//TRIPOS_TYPE_C_3_H3       12
    PLP_CLASH_CLASS_3,//TRIPOS_TYPE_C_ar         13
    PLP_CLASH_CLASS_3,//TRIPOS_TYPE_C_ar_H1      14
    PLP_CLASH_CLASS_3,//TRIPOS_TYPE_Ca           15
    PLP_CLASH_CLASS_1,//TRIPOS_TYPE_Cl           16
    PLP_CLASH_CLASS_3,//TRIPOS_TYPE_Du           17
    PLP_CLASH_CLASS_1,//TRIPOS_TYPE_F            18
    PLP_CLASH_CLASS_3,//TRIPOS_TYPE_H            19
    PLP_CLASH_CLASS_3,//TRIPOS_TYPE_H_P          20
    PLP_CLASH_CLASS_3,//TRIPOS_TYPE_I            21
    PLP_CLASH_CLASS_3,//TRIPOS_TYPE_K            22
    PLP_CLASH_CLASS_3,//TRIPOS_TYPE_Li           23
    PLP_CLASH_CLASS_3,//TRIPOS_TYPE_LP           24
    PLP_CLASH_CLASS_2,//TRIPOS_TYPE_N_1          25
    PLP_CLASH_CLASS_2,//TRIPOS_TYPE_N_2          26
    PLP_CLASH_CLASS_1,//TRIPOS_TYPE_N_3          27
    PLP_CLASH_CLASS_1,//TRIPOS_TYPE_N_4          28
    PLP_CLASH_CLASS_1,//TRIPOS_TYPE_N_am         29
    PLP_CLASH_CLASS_2,//TRIPOS_TYPE_N_ar         30
    PLP_CLASH_CLASS_1,//TRIPOS_TYPE_N_pl3        31
    PLP_CLASH_CLASS_3,//TRIPOS_TYPE_Na           32
    PLP_CLASH_CLASS_2,//TRIPOS_TYPE_O_2          33
    PLP_CLASH_CLASS_1,//TRIPOS_TYPE_O_3          34
    PLP_CLASH_CLASS_1,//TRIPOS_TYPE_O_co2        35
    PLP_CLASH_CLASS_3,//TRIPOS_TYPE_P_3          36
    PLP_CLASH_CLASS_3,//TRIPOS_TYPE_S_2          37
    PLP_CLASH_CLASS_3,//TRIPOS_TYPE_S_3          38
    PLP_CLASH_CLASS_3,//TRIPOS_TYPE_S_o          39
    PLP_CLASH_CLASS_3,//TRIPOS_TYPE_S_o2         40
    PLP_CLASH_CLASS_3,//TRIPOS_TYPE_Si           41
    PLP_CLASH_CLASS_3//TRIPOS_TYPE_MAXTYPES 42
};

#endif
