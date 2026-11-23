//+------------------------------------------------------------------+
//| FXSeasonalityEA.mq5                                              |
//| FX intraday seasonality portfolio (FXS 09/10): 12 time-of-day    |
//| signals, time entry and time exit, every trading day.            |
//| Repo: igarpet/equity-trading, tools/mt5. Version 0.1.0           |
//+------------------------------------------------------------------+
#property copyright "equity-trading"
#property version   "0.10"
#property description "12-signal FX time-of-day portfolio (FXS 09/10). Hedging account required."

#include <Trade/Trade.mqh>

//--- enums
enum ENUM_SERVER_CLOCK
  {
   SERVER_NY_PLUS_7 = 0,   // Server = New York + 7h (GMT+2 winter / GMT+3 US summer)
   SERVER_FIXED_GMT = 1    // Server = GMT + ServerGmtOffsetHours, no DST
  };

enum ENUM_SIGNAL_SET
  {
   SET_ALL12    = 0,       // ALL 12 (FXS 09/10, best tradable CAGR)
   SET_PREV_NEW = 1,       // PREVIOUS + NEW (8, FXS 07 "ALL")
   SET_PREVIOUS = 2,       // PREVIOUS (3)
   SET_NEW      = 3,       // NEW from FXS 06 (6)
   SET_NEW08    = 4        // NEW-08 crosses (4)
  };

enum ENUM_SIZE_BASE
  {
   SIZE_BALANCE = 0,       // Balance at entry
   SIZE_EQUITY  = 1        // Equity at entry
  };

//--- inputs
input group "Portfolio"
input ENUM_SIGNAL_SET InpSet          = SET_ALL12;  // Signal set
input double          InpLeverage     = 10.0;       // L: notional per signal = L x balance
input string          InpDisable      = "";         // Disable signals, e.g. "N-EJ-2,CADJPY-N1"
input string          InpMultipliers  = "";         // Per-signal size multipliers, e.g. "N-UJ-1=1.5;EURUSD-02=0.5"
input string          InpGotobiRules  = "";         // Per-signal gotobi rule: G=only gotobi, O=only other days, e.g. "USDJPY-01=G"
input ENUM_SIZE_BASE  InpSizeBase     = SIZE_BALANCE;
input double          InpMarginUse    = 0.95;       // Max share of free margin one entry may use

input group "Clock and symbols"
input ENUM_SERVER_CLOCK InpServerClock      = SERVER_NY_PLUS_7;
input int               InpServerGmtOffset  = 2;    // Used only with SERVER_FIXED_GMT
input string            InpSymbolPrefix     = "";
input string            InpSymbolSuffix     = "";   // e.g. ".r" if the broker uses suffixes

input group "Execution"
input int    InpMaxEntryDelaySec = 60;    // Skip the entry if it cannot be sent within this many seconds
input double InpMaxSpreadPips    = 0.0;   // Skip entry if spread is wider (0 = off)
input bool   InpSkipHolidays     = true;  // Skip day labels Dec 25 and Jan 1
input long   InpMagicBase        = 7310000;
input int    InpSlippagePoints   = 30;

input group "Logging"
input bool   InpWriteJournal     = true;  // Trade journal CSV in Terminal\Common\Files
input string InpJournalName      = "FXS_EA_trades.csv";

//--- signal table (from notebooks/FX_Intraday_Seasonality/Portfolio/09 summary.json)
struct Signal
  {
   string   id;
   string   pair;
   bool     ny;          // true: NY clock (minutes from 17:00 New York); false: UTC clock
   int      start_min;
   int      end_min;
   int      dir;         // +1 buy, -1 sell
   bool     on;
   double   mult;
   int      gotobi;      // 0 both days, 1 only gotobi, 2 only other days
   string   sym;
   long     magic;
   datetime last_label;  // clock day label of the last entry
   datetime exit_at;     // server time of the scheduled exit, 0 = flat
   int      trades;
   double   net_usd;
   double   net_pips;
  };

Signal   S[12];
CTrade   trade;
int      jp_hol[];
int      journal = INVALID_HANDLE;

//+------------------------------------------------------------------+
void SetSig(int i, string id, string pair, bool ny, int s, int e, int d)
  {
   S[i].id = id; S[i].pair = pair; S[i].ny = ny; S[i].start_min = s; S[i].end_min = e; S[i].dir = d;
   S[i].on = false; S[i].mult = 1.0; S[i].gotobi = 0; S[i].sym = InpSymbolPrefix + pair + InpSymbolSuffix;
   S[i].magic = InpMagicBase + i; S[i].last_label = 0; S[i].exit_at = 0;
   S[i].trades = 0; S[i].net_usd = 0.0; S[i].net_pips = 0.0;
  }

void BuildSignals()
  {
   SetSig(0,  "EURUSD-02", "EURUSD", true,  660,  900,  -1);
   SetSig(1,  "USDJPY-01", "USDJPY", false, 55,   115,  -1);
   SetSig(2,  "USDJPY-02", "USDJPY", false, 25,   55,   +1);
   SetSig(3,  "N-EU-1",    "EURUSD", true,  885,  900,  -1);
   SetSig(4,  "N-UJ-1",    "USDJPY", false, 55,   70,   -1);
   SetSig(5,  "N-UJ-3",    "USDJPY", true,  1340, 1400, +1);
   SetSig(6,  "N-EJ-1",    "EURJPY", true,  1315, 1435, +1);
   SetSig(7,  "N-EJ-2",    "EURJPY", true,  780,  900,  -1);
   SetSig(8,  "AUDJPY-N1", "AUDJPY", true,  1195, 1435, +1);
   SetSig(9,  "AUDJPY-N2", "AUDJPY", false, 60,   180,  +1);
   SetSig(10, "CADJPY-N1", "CADJPY", true,  1070, 1310, +1);
   SetSig(11, "GBPJPY-N1", "GBPJPY", true,  1195, 1435, +1);

   int prev[] = {0, 1, 2};
   int nw[]   = {3, 4, 2, 5, 6, 7};
   int n08[]  = {8, 9, 10, 11};
   switch(InpSet)
     {
      case SET_ALL12:    for(int i = 0; i < 12; i++) S[i].on = true; break;
      case SET_PREV_NEW: for(int i = 0; i < 8; i++)  S[i].on = true; break;
      case SET_PREVIOUS: for(int i = 0; i < ArraySize(prev); i++) S[prev[i]].on = true; break;
      case SET_NEW:      for(int i = 0; i < ArraySize(nw); i++)   S[nw[i]].on = true; break;
      case SET_NEW08:    for(int i = 0; i < ArraySize(n08); i++)  S[n08[i]].on = true; break;
     }
  }

int FindSig(string id)
  {
   string u = id; StringTrimLeft(u); StringTrimRight(u); StringToUpper(u);
   for(int i = 0; i < 12; i++)
     {
      string v = S[i].id; StringToUpper(v);
      if(v == u) return i;
     }
   return -1;
  }

bool ParseLists()
  {
   string parts[];
   int n = StringSplit(InpDisable, ',', parts);
   for(int k = 0; k < n; k++)
     {
      string p = parts[k]; StringTrimLeft(p); StringTrimRight(p);
      if(p == "") continue;
      int i = FindSig(p);
      if(i < 0) { PrintFormat("Unknown signal in InpDisable: %s", p); return false; }
      S[i].on = false;
     }
   n = StringSplit(InpMultipliers, ';', parts);
   for(int k = 0; k < n; k++)
     {
      string kv[];
      if(StringSplit(parts[k], '=', kv) != 2) continue;
      int i = FindSig(kv[0]);
      if(i < 0) { PrintFormat("Unknown signal in InpMultipliers: %s", kv[0]); return false; }
      S[i].mult = StringToDouble(kv[1]);
     }
   n = StringSplit(InpGotobiRules, ';', parts);
   for(int k = 0; k < n; k++)
     {
      string kv[];
      if(StringSplit(parts[k], '=', kv) != 2) continue;
      int i = FindSig(kv[0]);
      if(i < 0) { PrintFormat("Unknown signal in InpGotobiRules: %s", kv[0]); return false; }
      string r = kv[1]; StringTrimLeft(r); StringTrimRight(r); StringToUpper(r);
      if(r == "G") S[i].gotobi = 1;
      else if(r == "O") S[i].gotobi = 2;
      else if(r == "B") S[i].gotobi = 0;
      else { PrintFormat("Gotobi rule must be G, O or B: %s", parts[k]); return false; }
     }
   return true;
  }

//+------------------------------------------------------------------+
//| Calendar and clocks                                              |
//+------------------------------------------------------------------+
datetime MakeDate(int y, int m, int d)
  {
   MqlDateTime t; ZeroMemory(t);
   t.year = y; t.mon = m; t.day = d;
   return StructToTime(t);
  }

datetime DayStart(datetime t) { return t - (t % 86400); }

int Ymd(datetime t) { MqlDateTime s; TimeToStruct(t, s); return s.year * 10000 + s.mon * 100 + s.day; }

int Dow(datetime t) { MqlDateTime s; TimeToStruct(t, s); return s.day_of_week; }   // 0 = Sunday

datetime NthSunday(int y, int m, int n)
  {
   datetime first = MakeDate(y, m, 1);
   int dow = Dow(first);
   return first + ((7 - dow) % 7) * 86400 + (n - 1) * 7 * 86400;
  }

// US daylight saving at UTC instant t: second Sunday of March 07:00 UTC to first Sunday of November 06:00 UTC.
bool UsDst(datetime utc)
  {
   MqlDateTime s; TimeToStruct(utc, s);
   datetime a = NthSunday(s.year, 3, 2) + 7 * 3600;
   datetime b = NthSunday(s.year, 11, 1) + 6 * 3600;
   return utc >= a && utc < b;
  }

datetime ServerToUtc(datetime srv)
  {
   if(InpServerClock == SERVER_FIXED_GMT) return srv - InpServerGmtOffset * 3600;
   datetime u = srv - 3 * 3600;                 // US summer: server = UTC+3
   if(UsDst(u)) return u;
   return srv - 2 * 3600;                       // US winter: server = UTC+2
  }

// NY clock: New York local time + 7h, so 17:00 New York = 00:00 of the day label (as the study's panels).
datetime NyClock(datetime utc) { return utc - (UsDst(utc) ? 4 : 5) * 3600 + 7 * 3600; }

datetime SignalClock(int i, datetime utc) { return S[i].ny ? NyClock(utc) : utc; }

//--- Japanese bank days and gotobi (GTF definition, gotobi.py 0.2.0)
string JpHolidayList()
  {
   return
"20150101,20150112,20150211,20150321,20150429,20150503,20150504,20150505,20150506,20150720,20150921,20150922," +
   "20150923,20151012,20151103,20151123,20151223,20160101,20160111,20160211,20160320,20160321,20160429,20160503," +
   "20160504,20160505,20160718,20160811,20160919,20160922,20161010,20161103,20161123,20161223,20170101,20170102," +
   "20170109,20170211,20170320,20170429,20170503,20170504,20170505,20170717,20170811,20170918,20170923,20171009," +
   "20171103,20171123,20171223,20180101,20180108,20180211,20180212,20180321,20180429,20180430,20180503,20180504," +
   "20180505,20180716,20180811,20180917,20180923,20180924,20181008,20181103,20181123,20181223,20181224,20190101," +
   "20190114,20190211,20190321,20190429,20190430,20190501,20190502,20190503,20190504,20190505,20190506,20190715," +
   "20190811,20190812,20190916,20190923,20191014,20191022,20191103,20191104,20191123,20200101,20200113,20200211," +
   "20200223,20200224,20200320,20200429,20200503,20200504,20200505,20200506,20200723,20200724,20200810,20200921," +
   "20200922,20201103,20201123,20210101,20210111,20210211,20210223,20210320,20210429,20210503,20210504,20210505," +
   "20210722,20210723,20210808,20210809,20210920,20210923,20211103,20211123,20220101,20220110,20220211,20220223," +
   "20220321,20220429,20220503,20220504,20220505,20220718,20220811,20220919,20220923,20221010,20221103,20221123," +
   "20230101,20230102,20230109,20230211,20230223,20230321,20230429,20230503,20230504,20230505,20230717,20230811," +
   "20230918,20230923,20231009,20231103,20231123,20240101,20240108,20240211,20240212,20240223,20240320,20240429," +
   "20240503,20240504,20240505,20240506,20240715,20240811,20240812,20240916,20240922,20240923,20241014,20241103," +
   "20241104,20241123,20250101,20250113,20250211,20250223,20250224,20250320,20250429,20250503,20250504,20250505," +
   "20250506,20250721,20250811,20250915,20250923,20251013,20251103,20251123,20251124,20260101,20260112,20260211," +
   "20260223,20260320,20260429,20260503,20260504,20260505,20260506,20260720,20260811,20260921,20260922,20260923," +
   "20261012,20261103,20261123,20270101,20270111,20270211,20270223,20270321,20270322,20270429,20270503,20270504," +
   "20270505,20270719,20270811,20270920,20270923,20271011,20271103,20271123";
  }

void LoadJpHolidays()
  {
   string parts[];
   int n = StringSplit(JpHolidayList(), ',', parts);
   ArrayResize(jp_hol, 0);
   for(int k = 0; k < n; k++)
     {
      if(StringLen(parts[k]) != 8) continue;
      int sz = ArraySize(jp_hol);
      ArrayResize(jp_hol, sz + 1);
      jp_hol[sz] = (int)StringToInteger(parts[k]);
     }
   ArraySort(jp_hol);
  }

bool IsJpHoliday(datetime d)
  {
   int v = Ymd(d);
   int lo = 0, hi = ArraySize(jp_hol) - 1;
   while(lo <= hi)
     {
      int mid = (lo + hi) / 2;
      if(jp_hol[mid] == v) return true;
      if(jp_hol[mid] < v) lo = mid + 1; else hi = mid - 1;
     }
   return false;
  }

bool IsJpBankDay(datetime d)
  {
   MqlDateTime s; TimeToStruct(d, s);
   if(s.day_of_week == 0 || s.day_of_week == 6) return false;
   if((s.mon == 12 && s.day == 31) || (s.mon == 1 && s.day <= 3)) return false;
   return !IsJpHoliday(d);
  }

datetime PrevBankDay(datetime d)
  {
   while(!IsJpBankDay(d)) d -= 86400;
   return d;
  }

int DaysInMonth(int y, int m)
  {
   int ny = (m == 12) ? y + 1 : y, nm = (m == 12) ? 1 : m + 1;
   return (int)((MakeDate(ny, nm, 1) - MakeDate(y, m, 1)) / 86400);
  }

// tokyo_day: 00:00 of a Tokyo calendar date. Gotobi = days 5,10,...,30 of a month rolled to the previous bank day.
bool IsGotobi(datetime tokyo_day)
  {
   MqlDateTime s; TimeToStruct(tokyo_day, s);
   for(int k = 0; k < 2; k++)
     {
      int y = s.year, m = s.mon + k;
      if(m > 12) { m = 1; y++; }
      int last = DaysInMonth(y, m);
      for(int dd = 5; dd <= 30; dd += 5)
        {
         if(dd > last) continue;
         if(PrevBankDay(MakeDate(y, m, dd)) == tokyo_day) return true;
        }
     }
   return false;
  }

//+------------------------------------------------------------------+
//| Positions and sizing                                             |
//+------------------------------------------------------------------+
double PipSize(string sym)
  {
   int digits = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);
   double pt = SymbolInfoDouble(sym, SYMBOL_POINT);
   return (digits == 3 || digits == 5) ? pt * 10.0 : pt;
  }

int CountPositions(int i)
  {
   int n = 0;
   for(int k = PositionsTotal() - 1; k >= 0; k--)
     {
      ulong t = PositionGetTicket(k);
      if(t == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) == S[i].magic && PositionGetString(POSITION_SYMBOL) == S[i].sym) n++;
     }
   return n;
  }

// Notional in account currency of one lot: price x tick value / tick size (no other symbol needed).
double NotionalPerLot(string sym)
  {
   double tv = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE);
   double ts = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE);
   double px = SymbolInfoDouble(sym, SYMBOL_BID);
   if(tv <= 0 || ts <= 0 || px <= 0) return 0.0;
   return px * tv / ts;
  }

double TargetLots(int i)
  {
   double base = (InpSizeBase == SIZE_EQUITY) ? AccountInfoDouble(ACCOUNT_EQUITY) : AccountInfoDouble(ACCOUNT_BALANCE);
   double npl = NotionalPerLot(S[i].sym);
   if(npl <= 0) return 0.0;
   double lots = base * InpLeverage * S[i].mult / npl;
   double step = SymbolInfoDouble(S[i].sym, SYMBOL_VOLUME_STEP);
   double vmin = SymbolInfoDouble(S[i].sym, SYMBOL_VOLUME_MIN);
   // margin cap
   ENUM_ORDER_TYPE ot = (S[i].dir > 0) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   double px = (S[i].dir > 0) ? SymbolInfoDouble(S[i].sym, SYMBOL_ASK) : SymbolInfoDouble(S[i].sym, SYMBOL_BID);
   double m1 = 0.0;
   if(OrderCalcMargin(ot, S[i].sym, 1.0, px, m1) && m1 > 0)
     {
      double cap = InpMarginUse * AccountInfoDouble(ACCOUNT_MARGIN_FREE) / m1;
      if(lots > cap)
        {
         PrintFormat("%s: size cut by free margin, %.2f -> %.2f lots", S[i].id, lots, cap);
         lots = cap;
        }
     }
   lots = MathFloor(lots / step + 1e-9) * step;
   if(lots < vmin) return 0.0;
   return NormalizeDouble(lots, 2);
  }

bool OpenSignal(int i)
  {
   double lots = TargetLots(i);
   if(lots <= 0) { PrintFormat("%s: size below minimum volume, skipped", S[i].id); return false; }
   double vmax = SymbolInfoDouble(S[i].sym, SYMBOL_VOLUME_MAX);
   double step = SymbolInfoDouble(S[i].sym, SYMBOL_VOLUME_STEP);
   trade.SetExpertMagicNumber(S[i].magic);
   trade.SetTypeFillingBySymbol(S[i].sym);
   bool any = false;
   double left = lots;
   while(left >= step - 1e-9)
     {
      double v = NormalizeDouble(MathMin(left, vmax), 2);
      bool ok = (S[i].dir > 0) ? trade.Buy(v, S[i].sym, 0, 0, 0, S[i].id)
                               : trade.Sell(v, S[i].sym, 0, 0, 0, S[i].id);
      if(!ok || (trade.ResultRetcode() != TRADE_RETCODE_DONE && trade.ResultRetcode() != TRADE_RETCODE_PLACED))
        {
         PrintFormat("%s: order failed, retcode %u %s", S[i].id, trade.ResultRetcode(), trade.ResultRetcodeDescription());
         break;
        }
      any = true;
      left -= v;
     }
   return any;
  }

void JournalPosition(int i, ulong pos_id)
  {
   if(!HistorySelectByPosition(pos_id)) return;
   double profit = 0, comm = 0, swap = 0, vol = 0, p_in = 0, p_out = 0;
   datetime t_in = 0, t_out = 0;
   for(int k = 0; k < HistoryDealsTotal(); k++)
     {
      ulong d = HistoryDealGetTicket(k);
      profit += HistoryDealGetDouble(d, DEAL_PROFIT);
      comm   += HistoryDealGetDouble(d, DEAL_COMMISSION);
      swap   += HistoryDealGetDouble(d, DEAL_SWAP);
      long e = HistoryDealGetInteger(d, DEAL_ENTRY);
      if(e == DEAL_ENTRY_IN)  { p_in = HistoryDealGetDouble(d, DEAL_PRICE); t_in = (datetime)HistoryDealGetInteger(d, DEAL_TIME); vol = HistoryDealGetDouble(d, DEAL_VOLUME); }
      if(e == DEAL_ENTRY_OUT) { p_out = HistoryDealGetDouble(d, DEAL_PRICE); t_out = (datetime)HistoryDealGetInteger(d, DEAL_TIME); }
     }
   double pips = (p_out - p_in) * S[i].dir / PipSize(S[i].sym);
   double net = profit + comm + swap;
   S[i].trades++; S[i].net_usd += net; S[i].net_pips += pips;
   if(journal != INVALID_HANDLE)
      FileWrite(journal, S[i].id, S[i].sym, S[i].dir, DoubleToString(vol, 2),
                TimeToString(t_in, TIME_DATE | TIME_SECONDS), DoubleToString(p_in, 5),
                TimeToString(t_out, TIME_DATE | TIME_SECONDS), DoubleToString(p_out, 5),
                DoubleToString(pips, 2), DoubleToString(profit, 2), DoubleToString(comm, 2), DoubleToString(swap, 2),
                DoubleToString(net, 2), DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE), 2));
  }

void CloseSignal(int i)
  {
   trade.SetExpertMagicNumber(S[i].magic);
   for(int k = PositionsTotal() - 1; k >= 0; k--)
     {
      ulong t = PositionGetTicket(k);
      if(t == 0) continue;
      if(PositionGetInteger(POSITION_MAGIC) != S[i].magic || PositionGetString(POSITION_SYMBOL) != S[i].sym) continue;
      ulong pid = (ulong)PositionGetInteger(POSITION_IDENTIFIER);
      if(trade.PositionClose(t, InpSlippagePoints))
         JournalPosition(i, pid);
      else
         PrintFormat("%s: close failed, retcode %u %s", S[i].id, trade.ResultRetcode(), trade.ResultRetcodeDescription());
     }
  }

string GvName(int i) { return StringFormat("FXS_%I64d_exit", S[i].magic); }

//+------------------------------------------------------------------+
//| Scheduler                                                        |
//+------------------------------------------------------------------+
void Step()
  {
   datetime srv = MQLInfoInteger(MQL_TESTER) ? TimeCurrent() : TimeTradeServer();
   datetime utc = ServerToUtc(srv);
   for(int i = 0; i < 12; i++)
     {
      if(!S[i].on && S[i].exit_at == 0) continue;
      // exit
      if(S[i].exit_at > 0)
        {
         if(srv >= S[i].exit_at)
           {
            CloseSignal(i);
            if(CountPositions(i) == 0)
              {
               S[i].exit_at = 0;
               GlobalVariableDel(GvName(i));
              }
           }
         continue;
        }
      if(!S[i].on) continue;
      // entry
      datetime clk = SignalClock(i, utc);
      datetime label = DayStart(clk);
      if(label == S[i].last_label) continue;
      datetime entry_clk = label + S[i].start_min * 60;
      if(clk < entry_clk || clk >= entry_clk + InpMaxEntryDelaySec) continue;
      int dow = Dow(label);
      if(dow == 0 || dow == 6) { S[i].last_label = label; continue; }
      if(InpSkipHolidays)
        {
         MqlDateTime l; TimeToStruct(label, l);
         if((l.mon == 12 && l.day == 25) || (l.mon == 1 && l.day == 1)) { S[i].last_label = label; continue; }
        }
      if(S[i].gotobi != 0)
        {
         datetime entry_utc = utc - (clk - entry_clk);
         bool g = IsGotobi(DayStart(entry_utc + 9 * 3600));
         if((S[i].gotobi == 1 && !g) || (S[i].gotobi == 2 && g)) { S[i].last_label = label; continue; }
        }
      if(InpMaxSpreadPips > 0)
        {
         double spr = (SymbolInfoDouble(S[i].sym, SYMBOL_ASK) - SymbolInfoDouble(S[i].sym, SYMBOL_BID)) / PipSize(S[i].sym);
         if(spr > InpMaxSpreadPips) continue;          // retried until the entry window closes
        }
      if(OpenSignal(i))
        {
         S[i].last_label = label;
         S[i].exit_at = srv - (clk - entry_clk) + (S[i].end_min - S[i].start_min) * 60;
         GlobalVariableSet(GvName(i), (double)S[i].exit_at);
        }
     }
  }

//+------------------------------------------------------------------+
int OnInit()
  {
   BuildSignals();
   if(!ParseLists()) return INIT_PARAMETERS_INCORRECT;
   LoadJpHolidays();
   if(AccountInfoInteger(ACCOUNT_MARGIN_MODE) != ACCOUNT_MARGIN_MODE_RETAIL_HEDGING)
     {
      Print("FXSeasonalityEA needs a hedging account: USDJPY-01 and N-UJ-1 overlap on the same symbol.");
      return INIT_FAILED;
     }
   trade.SetDeviationInPoints(InpSlippagePoints);
   bool tester = (bool)MQLInfoInteger(MQL_TESTER);
   for(int i = 0; i < 12; i++)
     {
      if(!S[i].on) continue;
      if(!SymbolSelect(S[i].sym, true))
        {
         PrintFormat("Symbol %s not found. Set InpSymbolPrefix/InpSymbolSuffix.", S[i].sym);
         return INIT_FAILED;
        }
     }
   // recover scheduled exits after a restart (live/demo only)
   for(int i = 0; i < 12; i++)
     {
      if(!tester && GlobalVariableCheck(GvName(i)) && CountPositions(i) > 0)
        {
         S[i].exit_at = (datetime)GlobalVariableGet(GvName(i));
         PrintFormat("%s: open position recovered, exit at %s", S[i].id, TimeToString(S[i].exit_at));
        }
     }
   if(!tester)
     {
      long off = (long)(TimeTradeServer() - TimeGMT());
      long exp = (long)(TimeTradeServer() - ServerToUtc(TimeTradeServer()));
      if(MathAbs(off - exp) > 300)
         PrintFormat("WARNING: server GMT offset is %.2f h but InpServerClock implies %.2f h. Check InpServerClock.",
                     off / 3600.0, exp / 3600.0);
     }
   if(InpWriteJournal)
     {
      journal = FileOpen(InpJournalName, FILE_WRITE | FILE_CSV | FILE_COMMON | FILE_ANSI, ',');
      if(journal != INVALID_HANDLE)
         FileWrite(journal, "signal", "symbol", "dir", "lots", "open_time", "open_price", "close_time", "close_price",
                   "pips", "profit", "commission", "swap", "net", "balance");
     }
   string on = "";
   for(int i = 0; i < 12; i++)
      if(S[i].on) on += StringFormat("%s(x%.2f%s) ", S[i].id, S[i].mult, S[i].gotobi == 1 ? ",G" : (S[i].gotobi == 2 ? ",O" : ""));
   PrintFormat("FXSeasonalityEA 0.1.0: L=%.1f, signals: %s", InpLeverage, on);
   EventSetTimer(1);
   return INIT_SUCCEEDED;
  }

void OnDeinit(const int reason)
  {
   EventKillTimer();
   for(int i = 0; i < 12; i++)
      if(S[i].trades > 0)
         PrintFormat("%-10s trades %5d  net pips %9.1f  (%.3f/trade)  net USD %12.2f",
                     S[i].id, S[i].trades, S[i].net_pips, S[i].net_pips / S[i].trades, S[i].net_usd);
   if(journal != INVALID_HANDLE) { FileClose(journal); journal = INVALID_HANDLE; }
  }

void OnTimer() { Step(); }
void OnTick()  { Step(); }

double OnTester()
  {
   // custom criterion: CAGR over the test period
   double b0 = TesterStatistics(STAT_INITIAL_DEPOSIT);
   double b1 = b0 + TesterStatistics(STAT_PROFIT);
   if(b0 <= 0 || b1 <= 0) return -1.0;
   HistorySelect(0, TimeCurrent());
   datetime t0 = 0;
   if(HistoryDealsTotal() > 0) t0 = (datetime)HistoryDealGetInteger(HistoryDealGetTicket(0), DEAL_TIME);
   double yrs = (double)(TimeCurrent() - t0) / (365.25 * 86400.0);
   if(yrs <= 0) return 0.0;
   return MathPow(b1 / b0, 1.0 / yrs) - 1.0;
  }
//+------------------------------------------------------------------+
