"""
A broad, liquid candidate pool of US large/mid-cap stocks -- a good-faith
S&P-500-style snapshot, not a guaranteed live feed. Index membership
changes over time (additions, removals, spin-offs); this list will drift
slightly out of date eventually. It's meant as a big pool to price-filter
down from, not a claim of exact, current index membership.

main.py batch-prices this whole pool cheaply (a handful of API calls, not
one per symbol -- see tools.alpaca_tools.get_latest_prices) and filters it
down to whatever's actually affordable at the account's current equity
before running the full, expensive per-symbol pipeline on that subset.
"""

SP500_CANDIDATES = [
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL",
    "CRM", "CSCO", "AMD", "INTC", "QCOM", "TXN", "IBM", "NOW", "INTU", "AMAT",
    "ADBE", "MU", "LRCX", "KLAC", "SNPS", "CDNS", "ADI", "PANW", "CRWD", "FTNT",
    "ANET", "MSI", "APH", "TEL", "GLW", "HPQ", "HPE", "DELL", "NTAP", "STX",
    "WDC", "ON", "MCHP", "SWKS", "QRVO", "ENPH", "FSLR", "TER", "ZBRA", "TDY",
    "KEYS", "GEN", "AKAM", "EPAM", "PTC", "TYL", "FFIV", "JNPR", "NXPI", "MPWR",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW", "USB",
    "PNC", "TFC", "COF", "BK", "STT", "AIG", "MET", "PRU", "ALL", "TRV",
    "PGR", "CB", "AFL", "AON", "MMC", "AJG", "WTW", "CME", "ICE", "NDAQ",
    "SPGI", "MCO", "FIS", "FI", "GPN", "SYF", "DFS", "RF", "CFG", "KEY",
    "HBAN", "FITB", "MTB", "ZION", "CMA", "NTRS", "IVZ", "BEN", "AMP", "RJF",
    "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "TMO", "ABT", "BMY", "CVS",
    "AMGN", "MDT", "DHR", "GILD", "ISRG", "SYK", "VRTX", "REGN", "CI", "ELV",
    "ZTS", "BSX", "HUM", "BDX", "EW", "IDXX", "A", "IQV", "MRNA", "BIIB",
    "HCA", "CNC", "MCK", "COR", "CAH", "DXCM", "ALGN", "RMD", "WAT", "STE",
    "PODD", "GEHC", "SOLV", "VTRS", "INCY", "MOH", "UHS", "DVA", "CRL", "TECH",
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "WMB",
    "KMI", "OKE", "HES", "BKR", "FANG", "TRGP", "HAL", "DVN", "CTRA", "EQT",
    "APA", "MRO",
    "WMT", "PG", "KO", "PEP", "COST", "NKE", "MCD", "SBUX", "TGT", "HD",
    "LOW", "BKNG", "TJX", "CMG", "ORLY", "AZO", "ROST", "YUM", "MAR", "HLT",
    "DHI", "LEN", "NVR", "PHM", "GM", "F", "APTV", "EBAY", "ETSY", "DPZ",
    "DG", "DLTR", "KMX", "BBY", "GPC", "TSCO", "ULTA", "LULU", "RL", "PVH",
    "MO", "PM", "CL", "KMB", "GIS", "K", "HSY", "MDLZ", "STZ", "TAP",
    "MNST", "KDP", "SYY", "KR", "ADM", "TSN", "HRL", "CAG", "CHD", "CLX",
    "EL", "KVUE",
    "BA", "CAT", "GE", "HON", "UPS", "LMT", "MMM", "DE", "RTX", "UNP",
    "NOC", "GD", "CSX", "NSC", "FDX", "WM", "EMR", "ETN", "ITW", "PH",
    "CMI", "ROK", "DOV", "XYL", "AME", "IR", "PCAR", "PWR", "JCI", "CARR",
    "OTIS", "TT", "FAST", "PAYX", "VRSK", "CTAS", "URI", "GWW", "LHX", "TXT",
    "HWM", "BR", "J", "IEX", "SNA", "EFX",
    "DIS", "CMCSA", "VZ", "T", "NFLX", "TMUS", "CHTR", "EA", "TTWO", "WBD",
    "OMC", "IPG", "LYV", "MTCH", "PARA",
    "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "PEG", "ED",
    "WEC", "ES", "AWK", "DTE", "PPL", "FE", "AEE", "CMS", "CNP", "ATO",
    "NI", "EVRG", "LNT",
    "LIN", "FCX", "SHW", "APD", "ECL", "NEM", "NUE", "DOW", "DD", "PPG",
    "VMC", "MLM", "IFF", "ALB", "CTVA", "CF", "MOS", "FMC", "STLD", "EMN",
    "AMT", "PLD", "EQIX", "PSA", "O", "WELL", "SPG", "DLR", "CCI", "AVB",
    "EQR", "VTR", "SBAC", "EXR", "MAA", "ESS", "UDR", "CPT", "ARE", "INVH",
    "KIM", "REG", "HST", "BXP",
    "PYPL", "SQ", "SHOP", "UBER", "LYFT", "ABNB", "DASH", "COIN", "RBLX", "SNOW",
    "DDOG", "NET", "ZS", "OKTA", "TWLO", "DOCU", "ZM", "PINS", "SNAP", "ROKU",
    "SPOT", "TTD", "APP", "U", "PATH", "BILL", "HUBS", "TEAM", "WDAY", "VEEV",
    "CDW", "GDDY", "PAYC", "SSNC", "FICO", "MKTX", "MSCI", "TROW", "AMCR", "SEE",
    "CE", "LYB", "WY", "PKG", "IP", "BALL", "CCK", "AVY", "SWK", "ALLE",
    "WBA", "COTY", "NWL", "HAS", "MHK", "WHR", "LEG", "HRB", "UAA",
    "CPB", "SJM", "LW", "BG", "INGR", "CAKE", "WING",
    "TXRH", "DRI", "BLMN", "EAT", "PLAY", "CBRL", "JACK", "SHAK", "CROX", "DECK",
    "SKX", "COLM", "VFC", "GPS", "ANF", "AEO", "URBN", "BURL", "FIVE",
    "OLLI", "BJ", "CASY", "MUSA", "SFM", "ACI", "GRPN", "CHWY",
    "W", "OSTK", "REAL", "CVNA", "LAD", "PAG", "GPI", "ABG", "SAH",
    "RUSHA", "AN", "AAP", "GT", "BWA", "LEA", "ADNT", "DAN", "MGA",
    "ALV", "VC", "LKQ", "GNTX", "STLA", "HMC", "TM",
    "BMRN", "ALNY", "SRPT", "EXEL", "IONS", "RARE", "ACAD", "SGEN",
    "JAZZ", "PCVX", "NBIX", "UTHR", "HALO", "SUPN", "LGND", "CRSP", "EDIT", "NTLA",
    "BEAM", "RXRX", "RCM", "OMCL", "HQY", "TDOC", "AMED", "ENSG", "ACHC", "OPCH",
    "USPH", "CHE", "SEM", "EHC", "ADUS", "PNTG",
    "ALLY", "SNV", "EWBC", "WBS", "PB", "FHN", "CFR", "GBCI", "UMBF",
    "ONB", "FFIN", "TCBI", "PPBI", "HOMB", "BANF", "NBTB", "CVBF", "FULT", "WSFS",
    "REXR", "FR", "STAG", "TRNO", "EGP", "COLD", "LSI", "NSA", "CUBE",
    "PSB", "IRM", "WPC", "NNN", "ADC", "EPRT", "GTY", "SRC", "FCPT", "STOR",
]
