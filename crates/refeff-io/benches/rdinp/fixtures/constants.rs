pub(crate) const FALLBACK_INPUT: &str = r#"
TITLE Cu crystal
EDGE K
SCF 5.0
CONTROL 1 1 1 1 1 1
PRINT 0 0 0 0 0 0
POTENTIALS
0 29 Cu
1 29 Cu
ATOMS
0.0 0.0 0.0 0 Cu0 0.0 0
1.805 1.805 0.0 1 Cu1 2.55266 1
-1.805 1.805 0.0 1 Cu1 2.55266 2
1.805 -1.805 0.0 1 Cu1 2.55266 3
-1.805 -1.805 0.0 1 Cu1 2.55266 4
END
"#;

pub(crate) const DENSITY_INPUT_BENCH: &str = r#"
line,line.dat,0.0,1.0,2.0,core
1.0,0.0,0.0,251
plane plane.dat 0.0, 1.0 2.0
1.0,0.0,0.0,101
0.0,1.0,0.0,101
volume volume.bin 0.0 0.0 0.0
1.0,0.0,0.0,41
0.0,1.0,0.0,41
0.0,0.0,1.0,41
"#;

pub(crate) const FULLSPECTRUM_OPTIONS_BENCH: &str = r#"
CONTROL 1 0 1 0 1 0
EGRID 5.0 120.0 230
DRUDE 1.5E-15 0.025
VALENCE
EELS
DETAIL
COMPONENT Cu2 29 0.0847 EDGES
K CONV
4 DETAIL
M1 BACKGROUND
COMPONENT O1 8 DETAIL
1
L1
"#;

pub(crate) const DMDW_ENABLED_INPUT_BENCH: &str = concat!(
    "   1\n",
    "   6\n",
    "   1    450.000\n",
    "   0\n",
    "feff.dym\n",
    "   1\n",
    "   2   1   0          29.78\n",
);

pub(crate) const DMDW_OUT_BENCH: &str = concat!(
    "# Lanczos recursion order:    6\n",
    "# Temperature:  450.00\n",
    "# Dynamical matrix file: feff.dym\n",
    "\n",
    "--------------------------------------------------------------\n",
    " Path Indices:    1   2\n",
    " PDOS Poles:\n",
    "     Freq. (THz)    Weight\n",
    "        2.860       0.039469598\n",
    "        3.854       0.182890396\n",
    "        4.940       0.220041663\n",
    "        6.026       0.159715119\n",
    "        6.812       0.284980130\n",
    "        7.306       0.112876736\n",
    "\n",
    " PDOS Einstein freq (single pole), associated temp and eff. force constant: \n",
    " Freq (THz)   Temp (K)   Eff. FC (N/m)\n",
    "   5.784       277.60      69.6914\n",
    "\n",
    " pDOS n Moments, associated Einstein freqs, temps and eff. force constants:\n",
    "  n     Mom (THz^n)   Freq (THz)     Temp (K)    Eff. FC (N/m)\n",
    " -2       0.03881       5.07607       243.60      53.6688\n",
    " -1       0.18959       5.27461       253.13      57.9492\n",
    "  0       0.99997     ---------     --------\n",
    "  1       5.63317       5.63317       270.34      66.0957\n",
    "  2      33.45823       5.78431       277.59      69.6899\n",
    "\n",
    " Path Red. Mass (AMU):   31.773000\n",
    " Path Length (Ang), s^2 (1e-3 Ang^2):  2.5323  11.8576\n",
    "--------------------------------------------------------------\n",
);

pub(crate) const EDGES_DAT_BENCH: &str = concat!(
    " # emu, M_kk, gam\n",
    "   330.31915602984373        1.0000000000000000        6.3546470930994858E-002\n",
);

pub(crate) const CHEMICAL_DAT_BENCH: &str =
    "   0.0000000000000000        0.0000000000000000       -7.7292787791436899     \n";

pub(crate) const EMESH_DAT_BENCH: &str = concat!(
    "# edge, bohr, edge*hart      -0.13880      0.52918     -3.77698\n",
    "# ispec, ik0      0     1\n",
    " # ie, em(ie)*hart, xk(ie)\n",
    "    1            -3.77698             0.00000\n",
    "    2            -3.73888             0.10000\n",
    "    3            -3.62458             0.20000\n",
    "    4            -3.43408             0.30000\n",
    "    5            -3.16738             0.40000\n",
);

pub(crate) const FPF0_DAT_BENCH: &str = concat!(
    "  atom Z =           29\n",
    "       -1.46689E-01       -8.39242E-02 total energy part of fprime - 5/3*E_tot/mc**2\n",
    "           5\n",
    "  2.00000    -332.657   1\n",
    "  0.00162     -36.320   3\n",
    "  0.00317     -35.556   4\n",
    "  0.00017      -3.431   6\n",
    "  0.00033      -3.329   7\n",
    "  0.0   29.0000\n",
    "  0.5   28.6430\n",
    "  1.0   27.7057\n",
    "  1.5   26.4437\n",
    "  2.0   25.0396\n",
    "  2.5   23.5793\n",
);

pub(crate) const MODULE_LOG_BENCH: &str = concat!(
    "Calculating SCF potentials ...\n",
    "FEFF-serial using 1 thread.\n",
    "Done with module: potentials.\n",
);

pub(crate) const GTR_DAT_BENCH: &str = concat!(
    "    -0.616104     0.031773     1.624106     1.081113\n",
    "    -0.558474     0.031773     0.550420     1.190721\n",
    "    -0.506332     0.031773     0.087675     0.846187\n",
    "    -0.459680     0.031773    -0.391425     0.869742\n",
);

pub(crate) const GTRL_DAT_BENCH: &str = concat!(
    "    1   -0.43309363E+00    0.87593454E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00   -0.22036467E+01    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.16590562E-01   -0.38225502E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.19196035E+01    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.30759355E-01\n",
    "    2   -0.39809006E+00    0.45318252E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00   -0.17369893E+01    0.00000000E+00    0.00000000E+00    0.00000000E+00   -0.35253677E-02   -0.16114870E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.32349476E+00    0.00000000E+00    0.00000000E+00    0.00000000E+00    0.24426693E-01\n",
);

pub(crate) const XSCORR_RAW_DAT_BENCH: &str = concat!(
    " Temperature (Hatree) = 0\n",
    " Electronic Temperature (eV) = 0\n",
    " xloss =   0.86458999999999986       eV\n",
    " efermi =   -3.7769771800000003       eV\n",
    " Number of poles = 0\n",
    " Omega(Hart)    Re CCHI     Im CCHI   1-Fermi   Re xmu0    Im xmu0\n",
    "  -0.1388013015E+000  -0.1629950000E-004   0.1152400000E-003   0.5000000000E+000  -0.3259900000E-004   0.2304800000E-003\n",
    "  -0.1374011587E+000  -0.1689833765E-004   0.1185582229E-003   0.5140178752E+000  -0.3287500000E-004   0.2306500000E-003\n",
);
