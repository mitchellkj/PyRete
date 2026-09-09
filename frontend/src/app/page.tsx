"use client";

import React, { useState, useEffect } from "react";
import {
  Laptop,
  Coins,
  Send,
  Loader2,
  CheckCircle2,
  AlertCircle,
  BrainCircuit,
  Terminal,
  RefreshCw,
  Search,
  Calculator,
  ChevronDown,
  ChevronUp,
} from "lucide-react";

interface Product {
  id: string;
  name: string;
  category: string;
  brand: string;
}

interface Currency {
  code: string;
  name: string;
  symbol: string;
  approx_rate: number;
}

interface AuditItem {
  step: number | null;
  kind: string;
  content: string;
}

interface ApiResponse {
  success: boolean;
  session_id?: string;
  product: string;
  currency: string;
  exchange_rate: number;
  query: string;
  final_answer: string;
  status: string;
  cached?: boolean;
  audit_facts: AuditItem[];
}

const DEFAULT_PRODUCTS: Product[] = [
  { id: "macbook_pro_14", name: 'MacBook Pro', category: "Laptop", brand: "Apple" },
  { id: "dell_xps_15", name: "Dell XPS 15", category: "Laptop", brand: "Dell" },
  { id: "lenovo_thinkpad_x1", name: "Lenovo ThinkPad X1 Carbon", category: "Laptop", brand: "Lenovo" },
  { id: "asus_rog_g14", name: "ASUS ROG Zephyrus G14", category: "Gaming Laptop", brand: "ASUS" },
  { id: "ipad_pro_m4", name: "Apple iPad Pro M4", category: "Tablet", brand: "Apple" },
  { id: "galaxy_tab_s9", name: "Samsung Galaxy Tab S9", category: "Tablet", brand: "Samsung" },
  { id: "hp_spectre_x360", name: "HP Spectre x360", category: "2-in-1 Laptop", brand: "HP" },
];

const DEFAULT_CURRENCIES: Currency[] = [
  { code: "EUR", name: "Euro", symbol: "€", approx_rate: 0.85 },
  { code: "GBP", name: "British Pound", symbol: "£", approx_rate: 0.77 },
  { code: "JPY", name: "Japanese Yen", symbol: "¥", approx_rate: 150.0 },
  { code: "CAD", name: "Canadian Dollar", symbol: "C$", approx_rate: 1.36 },
  { code: "AUD", name: "Australian Dollar", symbol: "A$", approx_rate: 1.52 },
  { code: "CHF", name: "Swiss Franc", symbol: "Fr", approx_rate: 0.88 },
  { code: "INR", name: "Indian Rupee", symbol: "₹", approx_rate: 83.5 },
];

export default function PricingDashboard() {
  const [products] = useState<Product[]>(DEFAULT_PRODUCTS);
  const [currencies] = useState<Currency[]>(DEFAULT_CURRENCIES);

  const [selectedProduct, setSelectedProduct] = useState<Product>(DEFAULT_PRODUCTS[0]);
  const [selectedCurrency, setSelectedCurrency] = useState<Currency>(DEFAULT_CURRENCIES[0]);
  const [exchangeRate, setExchangeRate] = useState<number>(DEFAULT_CURRENCIES[0].approx_rate);

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [response, setResponse] = useState<ApiResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [showAudit, setShowAudit] = useState<boolean>(true);
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0);
  const [sessionId, setSessionId] = useState<string>("");

  // Initialize or restore session identifier for working memory scope
  useEffect(() => {
    if (typeof window !== "undefined") {
      let sid = localStorage.getItem("pyrete_session_id");
      if (!sid) {
        sid = "sess-" + Math.random().toString(36).substring(2, 11);
        localStorage.setItem("pyrete_session_id", sid);
      }
      setSessionId(sid);
    }
  }, []);

  // Update exchange rate when currency changes
  const handleCurrencySelect = (curr: Currency) => {
    setSelectedCurrency(curr);
    setExchangeRate(curr.approx_rate);
  };

  // Check backend health
  useEffect(() => {
    const apiBase = process.env.NEXT_PUBLIC_API_URL || "";
    fetch(`${apiBase}/health`)
      .then((res) => (res.ok ? res.json() : Promise.reject()))
      .then(() => setBackendOnline(true))
      .catch(() => setBackendOnline(false));
  }, []);

  // Timer while loading
  useEffect(() => {
    let interval: any;
    if (isLoading) {
      setElapsedSeconds(0);
      interval = setInterval(() => {
        setElapsedSeconds((prev) => prev + 1);
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isLoading]);

  // Guard-railed query representation
  const synthesizedQuery =
    `What is the total retail purchase price (MSRP) of a new entry-level ${selectedProduct.name} in USD? ` +
    `Do not use monthly financing. ` +
    `How much would it cost in ${selectedCurrency.name} (${selectedCurrency.code}) ` +
    `if the exchange rate is ${exchangeRate} ${selectedCurrency.code} for 1 USD?`;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setResponse(null);

    try {
      const apiBase = process.env.NEXT_PUBLIC_API_URL || "";
      const res = await fetch(`${apiBase}/api/pricing`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          product: selectedProduct.name,
          currency_code: selectedCurrency.code,
          currency_name: selectedCurrency.name,
          exchange_rate: Number(exchangeRate),
          session_id: sessionId || undefined,
        }),
      });

      if (!res.ok) {
        throw new Error(`Backend error (${res.status}): Make sure the FastAPI server is running`);
      }

      const data: ApiResponse = await res.json();
      if (data.session_id && data.session_id !== sessionId) {
        setSessionId(data.session_id);
        if (typeof window !== "undefined") {
          localStorage.setItem("pyrete_session_id", data.session_id);
        }
      }
      setResponse(data);
    } catch (err: any) {
      setError(err.message || "Failed to reach backend");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="min-h-screen py-8 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
      {/* Header */}
      <header className="mb-8 border-b border-slate-200 pb-6">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="p-2 bg-emerald-600 text-white rounded-lg shadow-sm">
                <BrainCircuit className="w-6 h-6" />
              </span>
              <h1 className="text-2xl font-bold text-slate-900 tracking-tight">
                PyRete ReAct Hardware Pricing
              </h1>
            </div>
            <p className="text-sm text-slate-600 mt-1">
              Goal-Driven Forward-Chaining Rule Coordinator for Automated Price Discovery & Currency Conversion
            </p>
          </div>

          <div className="flex items-center gap-2 text-xs flex-wrap">
            {sessionId && (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full font-mono text-slate-700 bg-slate-100 border border-slate-200" title={`Session ID: ${sessionId}`}>
                <span className="w-2 h-2 rounded-full bg-blue-500" />
                Session: {sessionId.slice(0, 12)}
              </span>
            )}
            <span
              className={`inline-flex items-center px-2.5 py-1 rounded-full font-medium ${
                backendOnline
                  ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                  : backendOnline === false
                  ? "bg-rose-50 text-rose-700 border border-rose-200"
                  : "bg-slate-100 text-slate-600"
              }`}
            >
              <span
                className={`w-2 h-2 mr-1.5 rounded-full ${
                  backendOnline ? "bg-emerald-500" : backendOnline === false ? "bg-rose-500" : "bg-slate-400"
                }`}
              />
              {backendOnline ? "FastAPI Online" : backendOnline === false ? "FastAPI Offline" : "Checking..."}
            </span>
          </div>
        </div>
      </header>

      {/* Main Grid: Left Column Products | Right Column Currencies & Query */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Product Selection (5 cols) */}
        <section className="lg:col-span-5 space-y-4">
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-base font-semibold text-slate-900 flex items-center gap-2">
                <Laptop className="w-4 h-4 text-emerald-600" />
                1. Select Hardware Product
              </h2>
              <span className="text-xs text-slate-400 font-mono">7 options</span>
            </div>

            <div className="space-y-2">
              {products.map((p) => {
                const isSelected = selectedProduct.id === p.id;
                return (
                  <label
                    key={p.id}
                    onClick={() => setSelectedProduct(p)}
                    className={`flex items-center justify-between p-3 rounded-lg border cursor-pointer transition-all ${
                      isSelected
                        ? "border-emerald-500 bg-emerald-50/50 shadow-sm ring-1 ring-emerald-500/20"
                        : "border-slate-200 hover:border-slate-300 hover:bg-slate-50/50"
                    }`}
                  >
                    <div className="flex items-center gap-3">
                      <input
                        type="radio"
                        name="product"
                        checked={isSelected}
                        onChange={() => setSelectedProduct(p)}
                        className="h-4 w-4 text-emerald-600 focus:ring-emerald-500 border-slate-300"
                      />
                      <div>
                        <div className="text-sm font-medium text-slate-900">{p.name}</div>
                        <div className="text-xs text-slate-500">{p.category}</div>
                      </div>
                    </div>
                    <span className="text-xs px-2 py-0.5 bg-white rounded border border-slate-200 text-slate-600 font-mono">
                      {p.brand}
                    </span>
                  </label>
                );
              })}
            </div>
          </div>
        </section>

        {/* Right Column: Currency Selection, Query Preview & Output (7 cols) */}
        <section className="lg:col-span-7 space-y-6">
          {/* Currency Card */}
          <div className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-base font-semibold text-slate-900 flex items-center gap-2">
                <Coins className="w-4 h-4 text-emerald-600" />
                2. Target Currency Denomination
              </h2>
              <span className="text-xs text-slate-400 font-mono">7 options</span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              {currencies.map((c) => {
                const isSelected = selectedCurrency.code === c.code;
                return (
                  <button
                    type="button"
                    key={c.code}
                    onClick={() => handleCurrencySelect(c)}
                    className={`p-2.5 rounded-lg border text-left transition-all ${
                      isSelected
                        ? "border-emerald-500 bg-emerald-50/60 ring-1 ring-emerald-500/20"
                        : "border-slate-200 hover:border-slate-300 hover:bg-slate-50/50"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-slate-900 text-sm">{c.code}</span>
                      <span className="text-xs font-mono text-slate-400">{c.symbol}</span>
                    </div>
                    <div className="text-xs text-slate-500 truncate">{c.name}</div>
                  </button>
                );
              })}
            </div>

            {/* Exchange Rate Adjustment */}
            <div className="mt-4 pt-4 border-t border-slate-100 flex items-center justify-between gap-4">
              <span className="text-xs text-slate-600">
                Benchmark Rate for 1 USD:
              </span>
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono text-slate-500">{selectedCurrency.code} =</span>
                <input
                  type="number"
                  step="0.01"
                  value={exchangeRate}
                  onChange={(e) => setExchangeRate(parseFloat(e.target.value) || 0)}
                  className="w-24 px-2.5 py-1 text-sm border border-slate-300 rounded-md font-mono text-right focus:outline-none focus:ring-1 focus:ring-emerald-500"
                />
              </div>
            </div>
          </div>

          {/* Guard-railed Query Preview & Submit Form */}
          <form onSubmit={handleSubmit} className="bg-white p-5 rounded-xl border border-slate-200 shadow-sm space-y-4">
            <div>
              <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1.5">
                Synthesized Guard-Railed Query
              </label>
              <div className="p-3 bg-slate-50 rounded-lg border border-slate-200 text-xs text-slate-700 font-mono leading-relaxed">
                {synthesizedQuery}
              </div>
              <p className="text-[11px] text-slate-400 mt-1.5">
                🛡️ Query is automatically guard-railed to demand retail MSRP and exclude monthly financing.
              </p>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className={`w-full py-3 px-4 rounded-lg font-medium text-sm flex items-center justify-center gap-2 text-white transition-all shadow-sm ${
                isLoading
                  ? "bg-slate-400 cursor-not-allowed"
                  : "bg-emerald-600 hover:bg-emerald-700 active:scale-[0.99]"
              }`}
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Running PyRete Coordinator ({elapsedSeconds}s)...
                </>
              ) : (
                <>
                  <Send className="w-4 h-4" />
                  Run PyRete Pricing Coordinator
                </>
              )}
            </button>
          </form>

          {/* Error Message */}
          {error && (
            <div className="p-4 bg-rose-50 border border-rose-200 rounded-xl text-sm text-rose-800 flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-rose-600 flex-shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold">Execution Error</p>
                <p className="text-xs mt-0.5">{error}</p>
              </div>
            </div>
          )}

          {/* Results Display */}
          {response && (
            <div className="space-y-4">
              {/* Final Answer Card */}
              <div
                className={`p-5 rounded-xl border shadow-sm ${
                  response.success
                    ? "bg-emerald-50/50 border-emerald-300 ring-1 ring-emerald-400/20"
                    : "bg-amber-50/60 border-amber-300"
                }`}
              >
                <div className="flex items-center justify-between gap-2 mb-2 flex-wrap">
                  <div className="flex items-center gap-2">
                    {response.success ? (
                      <CheckCircle2 className="w-5 h-5 text-emerald-600" />
                    ) : (
                      <AlertCircle className="w-5 h-5 text-amber-600" />
                    )}
                    <h3 className="font-semibold text-slate-900 text-sm">
                      {response.success ? "Final Answer Result" : "Search Status"}
                    </h3>
                  </div>
                  {response.cached && (
                    <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-100 text-emerald-800 border border-emerald-200">
                      ⚡ Instant (Cached Working Memory)
                    </span>
                  )}
                </div>

                <div className="text-slate-800 text-sm leading-relaxed whitespace-pre-wrap font-medium">
                  {response.final_answer}
                </div>
              </div>

              {/* PyRete Working Memory Audit Accordion */}
              {response.audit_facts && response.audit_facts.length > 0 && (
                <div className="bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
                  <button
                    type="button"
                    onClick={() => setShowAudit(!showAudit)}
                    className="w-full px-5 py-3.5 bg-slate-50 border-b border-slate-200 flex items-center justify-between text-left hover:bg-slate-100/70 transition-colors"
                  >
                    <div className="flex items-center gap-2 text-xs font-semibold text-slate-700 uppercase tracking-wider">
                      <Terminal className="w-4 h-4 text-slate-500" />
                      PyRete Working Memory Audit ({response.audit_facts.length} Facts)
                    </div>
                    {showAudit ? (
                      <ChevronUp className="w-4 h-4 text-slate-500" />
                    ) : (
                      <ChevronDown className="w-4 h-4 text-slate-500" />
                    )}
                  </button>

                  {showAudit && (
                    <div className="p-4 space-y-2.5 max-h-96 overflow-y-auto font-mono text-xs">
                      {response.audit_facts.map((item, idx) => {
                        const isAction = item.kind === "Action";
                        const isObs = item.kind === "Observation";
                        const isFinal = item.kind === "FinalAnswer";
                        const isThought = item.kind === "Thought";

                        return (
                          <div
                            key={idx}
                            className={`p-2.5 rounded border ${
                              isAction
                                ? "bg-blue-50/70 border-blue-200 text-blue-900"
                                : isObs
                                ? "bg-slate-50 border-slate-200 text-slate-800"
                                : isFinal
                                ? "bg-emerald-50 border-emerald-200 text-emerald-900 font-semibold"
                                : "bg-purple-50/50 border-purple-200 text-purple-900"
                            }`}
                          >
                            <div className="flex items-center justify-between text-[11px] opacity-70 mb-1">
                              <span>
                                {item.step ? `Step ${item.step}` : "Global"} • {item.kind}
                              </span>
                              {isAction && item.content.includes("duckduck") && (
                                <span className="inline-flex items-center gap-1">
                                  <Search className="w-3 h-3" /> DDGS Tool
                                </span>
                              )}
                              {isAction && item.content.includes("Calculator") && (
                                <span className="inline-flex items-center gap-1">
                                  <Calculator className="w-3 h-3" /> Math Tool
                                </span>
                              )}
                            </div>
                            <div className="whitespace-pre-wrap break-words">{item.content}</div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </section>
      </div>
    </main>
  );
}
