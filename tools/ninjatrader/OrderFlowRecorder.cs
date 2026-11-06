// OrderFlowRecorder — NinjaTrader 8 indicator (Futures_Order_Flow, FOF 001 §5, Holdout B).
//
// Logs every trade print of the chart's instrument with the best bid/ask seen strictly before it,
// timestamped in UTC, one CSV per instrument per UTC day:
//   Documents\NinjaTrader 8\OrderFlowRecorder\<instrument>_<yyyy-MM-dd>.csv
//   ts_utc,instrument,price,size,bid,ask
//
// It places no orders and draws nothing. Real-time only: historical bars are ignored.
// Read with equity_trading.futures.read_recorder().
#region Using declarations
using System;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Globalization;
using System.IO;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
#endregion

namespace NinjaTrader.NinjaScript.Indicators
{
    public class OrderFlowRecorder : Indicator
    {
        private readonly object sync = new object();
        private StreamWriter writer;
        private string writerDay;
        private string folder;
        private double lastBid = double.NaN;
        private double lastAsk = double.NaN;
        private int sinceFlush;

        [NinjaScriptProperty]
        [Range(1, int.MaxValue)]
        [Display(Name = "Flush every N trades", Order = 1, GroupName = "Recorder")]
        public int FlushEvery { get; set; }

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name = "OrderFlowRecorder";
                Description = "Logs trades with the prior best bid/ask in UTC (Futures_Order_Flow Holdout B).";
                Calculate = Calculate.OnEachTick;
                IsOverlay = true;
                DisplayInDataBox = false;
                PaintPriceMarkers = false;
                IsSuspendedWhileInactive = false;   // keep recording when the chart tab is hidden
                FlushEvery = 200;
            }
            else if (State == State.DataLoaded)
            {
                folder = Path.Combine(NinjaTrader.Core.Globals.UserDataDir, "OrderFlowRecorder");
                Directory.CreateDirectory(folder);
            }
            else if (State == State.Terminated)
            {
                lock (sync) { CloseWriter(); }
            }
        }

        protected override void OnBarUpdate() { }

        protected override void OnMarketData(MarketDataEventArgs e)
        {
            if (State != State.Realtime)
                return;

            if (e.MarketDataType == MarketDataType.Bid) { lastBid = e.Price; return; }
            if (e.MarketDataType == MarketDataType.Ask) { lastAsk = e.Price; return; }
            if (e.MarketDataType != MarketDataType.Last) return;

            // NinjaTrader stamps events in the time zone of Tools > Options > General.
            DateTime utc = TimeZoneInfo.ConvertTimeToUtc(
                DateTime.SpecifyKind(e.Time, DateTimeKind.Unspecified),
                NinjaTrader.Core.Globals.GeneralOptions.TimeZoneInfo);

            string line = string.Join(",",
                utc.ToString("yyyy-MM-ddTHH:mm:ss.fffffff", CultureInfo.InvariantCulture) + "Z",
                Instrument.FullName,
                e.Price.ToString("R", CultureInfo.InvariantCulture),
                e.Volume.ToString(CultureInfo.InvariantCulture),
                double.IsNaN(lastBid) ? "" : lastBid.ToString("R", CultureInfo.InvariantCulture),
                double.IsNaN(lastAsk) ? "" : lastAsk.ToString("R", CultureInfo.InvariantCulture));

            lock (sync)
            {
                try
                {
                    EnsureWriter(utc);
                    writer.WriteLine(line);
                    if (++sinceFlush >= FlushEvery)
                    {
                        writer.Flush();
                        sinceFlush = 0;
                    }
                }
                catch (Exception ex)
                {
                    Print("OrderFlowRecorder: write failed: " + ex.Message);
                }
            }
        }

        private void EnsureWriter(DateTime utc)
        {
            string day = utc.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);
            if (writer != null && day == writerDay)
                return;
            CloseWriter();
            string name = Instrument.FullName.Replace(' ', '_').Replace('/', '-') + "_" + day + ".csv";
            string path = Path.Combine(folder, name);
            bool isNew = !File.Exists(path);
            writer = new StreamWriter(new FileStream(path, FileMode.Append, FileAccess.Write, FileShare.Read));
            if (isNew)
                writer.WriteLine("ts_utc,instrument,price,size,bid,ask");
            writerDay = day;
        }

        private void CloseWriter()
        {
            if (writer == null)
                return;
            try { writer.Flush(); writer.Dispose(); }
            catch (Exception) { }
            writer = null;
            sinceFlush = 0;
        }
    }
}
