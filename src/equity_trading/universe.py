"""Study universes. Fixed in advance; changes require a new study number."""

UNIVERSE_01 = {
    "tech": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "INTC", "CSCO"],
    "banks": ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "TFC", "SCHW"],
    "energy": ["XOM", "CVX", "COP", "EOG", "SLB", "PSX", "MPC", "VLO", "OXY", "HAL"],
    "utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ED", "PEG"],
}

# Select Sector SPDR ETF for each UNIVERSE_01 sector (used as the factor leg).
SECTOR_ETF_01 = {"tech": "XLK", "banks": "XLF", "energy": "XLE", "utilities": "XLU"}
