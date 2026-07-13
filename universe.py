"""
universe.py -- the tradeable universe.

Ported from the Kalman project's screener.py, with the 12 tickers its walk-forward run
proved are delisted/404 on Yahoo already removed (CTRA, MRO, CMA, ANSS, X, K, SKX, PARA,
FI, DFS, HES, SNV).

NOTE ON SURVIVORSHIP: this lists names liquid TODAY. Companies that went bankrupt or were
acquired are absent, which biases any backtest optimistically. Unfixable without a
point-in-time universe (CRSP). It is stated here so it is never forgotten.

Groups are kept only for reporting/diagnostics. Unlike the Kalman project, NOTHING in the
strategy selects on them -- every stock in the universe gets a signal, every day. That is
the whole point: no pair selection means no selection bias.
"""

UNIVERSE_US = {
    "country_etf": ["EWA", "EWC", "EWG", "EWU", "EWJ", "EWZ", "EWW", "EWH", "EWL", "EWP", "EWQ", "EWY", "EWT", "EWS", "EWM", "EWD", "EWI", "EWN", "INDA", "FXI", "EZA", "TUR"],
    "broad_etf": ["QQQ", "DIA", "IWM", "MDY", "VTI", "RSP"],
    "sector_etf": ["XLE", "XLF", "XLK", "XLV", "XLI", "XLP", "XLU", "XLB", "XLRE", "XLY", "XLC"],
    "industry_etf": ["SMH", "SOXX", "KBE", "KRE", "XBI", "IBB", "ITB", "XHB", "OIH", "XOP", "KIE", "XRT", "IYT", "JETS", "TAN", "ITA"],
    "commodity_etf": ["GLD", "SLV", "GDX", "GDXJ", "USO", "XME", "DBC", "UNG", "DBA", "CORN", "WEAT", "PPLT", "PALL", "COPX", "URA", "DBB"],
    "bond_etf": ["TLT", "IEF", "SHY", "LQD", "HYG", "AGG", "BND", "TIP", "MBB", "EMB", "JNK", "VCIT", "VCSH", "BIL"],
    "money_banks": ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "TFC", "BK", "STT"],
    "regional_banks": ["RF", "KEY", "CFG", "HBAN", "FITB", "MTB", "ZION", "ALLY", "FHN"],
    "oil_gas": ["XOM", "CVX", "COP", "EOG", "OXY", "PSX", "VLO", "MPC", "DVN", "FANG", "APA"],
    "oilfield_svcs": ["SLB", "HAL", "BKR", "NOV", "FTI"],
    "midstream": ["KMI", "WMB", "OKE", "ET", "EPD", "MPLX", "LNG", "TRGP"],
    "big_tech": ["MSFT", "AAPL", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA"],
    "software": ["ORCL", "CRM", "ADBE", "NOW", "INTU", "IBM", "CSCO", "SNOW", "WDAY", "TEAM", "DDOG", "PANW", "FTNT", "CRWD", "ZS", "NET", "ADSK"],
    "internet": ["NFLX", "DIS", "CMCSA", "WBD", "SPOT", "PINS", "SNAP", "UBER", "LYFT", "ABNB", "BKNG", "EXPE"],
    "semis": ["AMD", "INTC", "TXN", "ADI", "MU", "QCOM", "AVGO", "AMAT", "LRCX", "KLAC", "MRVL", "ON", "MCHP", "NXPI", "TSM", "ASML", "SWKS", "TER", "MPWR"],
    "payments": ["V", "MA", "AXP", "PYPL", "FIS", "GPN", "COF", "SYF"],
    "retail": ["WMT", "TGT", "COST", "HD", "LOW", "DG", "DLTR", "KR", "BJ", "ROST", "TJX", "BURL", "ULTA", "BBY", "TSCO"],
    "ecommerce": ["EBAY", "ETSY", "W", "CHWY", "SHOP", "MELI", "CVNA"],
    "pharma": ["JNJ", "PFE", "MRK", "ABBV", "BMY", "LLY", "GILD", "AMGN", "VRTX", "REGN", "BIIB", "ZTS"],
    "biotech": ["MRNA", "INCY", "BMRN", "ALNY", "EXEL", "SRPT", "NBIX"],
    "medical_dev": ["MDT", "ABT", "SYK", "BSX", "ISRG", "EW", "ZBH", "BDX", "DHR", "TMO", "A", "IDXX", "RMD", "DXCM", "ALGN"],
    "health_ins": ["UNH", "CI", "ELV", "HUM", "CNC", "CVS", "MOH"],
    "telecom": ["VZ", "T", "TMUS"],
    "staples": ["KO", "PEP", "PG", "CL", "MO", "PM", "GIS", "HSY", "KHC", "CAG", "MKC", "CLX", "KMB", "SYY", "STZ", "TAP", "KDP", "MDLZ", "ADM"],
    "autos": ["GM", "F", "TM", "STLA", "HMC"],
    "airlines": ["DAL", "UAL", "AAL", "LUV", "ALK"],
    "travel": ["MAR", "HLT", "RCL", "CCL", "NCLH", "MGM", "LVS", "WYNN", "H"],
    "utilities": ["DUK", "SO", "NEE", "D", "AEP", "EXC", "XEL", "ED", "WEC", "ES", "PEG", "SRE", "EIX", "AEE", "CMS", "DTE", "PPL", "FE", "ETR"],
    "industrials": ["CAT", "DE", "HON", "GE", "MMM", "EMR", "ETN", "ITW", "PH", "ROK", "DOV", "CMI", "PCAR", "GWW", "FAST", "IR", "AME"],
    "aero_defense": ["BA", "LMT", "RTX", "NOC", "GD", "LHX", "TDG", "HWM", "HII", "TXT"],
    "transports": ["UNP", "CSX", "NSC", "ODFL", "JBHT", "CHRW", "FDX", "UPS", "EXPD", "XPO"],
    "insurance": ["MET", "PRU", "AIG", "ALL", "TRV", "PGR", "CB", "HIG", "AFL", "CINF", "WRB", "ACGL", "L"],
    "asset_mgmt": ["BLK", "BX", "KKR", "APO", "BEN", "TROW", "IVZ", "AMP", "SCHW", "NTRS", "ICE", "CME", "NDAQ", "MCO", "SPGI", "MSCI"],
    "homebuilders": ["DHI", "LEN", "PHM", "NVR", "TOL", "KBH", "MHO", "BLDR", "MAS", "MHK"],
    "reits": ["AMT", "PLD", "CCI", "EQIX", "PSA", "O", "SPG", "DLR", "WELL", "VICI", "EQR", "AVB", "EXR", "MAA", "ESS", "ARE", "INVH", "KIM", "REG", "BXP"],
    "chemicals": ["LIN", "APD", "SHW", "ECL", "DD", "DOW", "PPG", "ALB", "LYB", "CE", "IFF", "EMN", "RPM"],
    "metals_mining": ["FCX", "NUE", "STLD", "CLF", "AA", "NEM", "GOLD", "RIO", "BHP", "VALE", "SCCO", "AEM"],
    "restaurants": ["MCD", "SBUX", "CMG", "YUM", "DRI", "DPZ", "QSR", "WEN", "TXRH", "CAKE"],
    "apparel": ["NKE", "LULU", "VFC", "RL", "TPR", "CPRI", "PVH", "UAA", "DECK", "CROX"],
    "agriculture": ["BG", "MOS", "CF", "NTR", "FMC", "CTVA"],
}


def all_tickers(universe=None):
    """Flat, de-duplicated, order-preserving ticker list."""
    universe = universe or UNIVERSE_US
    seen = []
    for ts in universe.values():
        for t in ts:
            if t not in seen:
                seen.append(t)
    return seen


def sector_map(universe=None):
    """ticker -> group (first group wins). Diagnostics only; never used for selection."""
    universe = universe or UNIVERSE_US
    m = {}
    for g, ts in universe.items():
        for t in ts:
            m.setdefault(t, g)
    return m
