import { useCallback, useEffect, useMemo, useState } from "react";

import {
  Activity,
  ArrowDownRight,
  ArrowRight,
  ArrowUpRight,
  BarChart3,
  BriefcaseBusiness,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleAlert,
  Clipboard,
  CloudUpload,
  Code2,
  FileCheck2,
  FileText,
  Gauge,
  HeartPulse,
  Home,
  Info,
  Lightbulb,
  LoaderCircle,
  Menu,
  RefreshCw,
  Search,
  Send,
  ShieldCheck,
  Sparkles,
  Target,
  Trash2,
  Upload,
  WandSparkles,
  X,
  Zap,
} from "lucide-react";

import "./index.css";

// --------------------------------------------------
// Configuration
// --------------------------------------------------

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const MAX_FILE_SIZE = 5 * 1024 * 1024;

const NAVIGATION = [
  { id: "home", label: "Dashboard", icon: Home },
  { id: "analyzer", label: "Resume Analyzer", icon: FileText },
  { id: "matcher", label: "Job Matcher", icon: Target },
  { id: "recommendations", label: "AI Recommendations", icon: Lightbulb },
];

const STEPS = [
  "Upload Resume",
  "Analyze Resume",
  "Job Description",
  "Match Results",
  "AI Recommendations",
];

// --------------------------------------------------
// Utility functions
// --------------------------------------------------

function getErrorMessage(data, fallback) {
  if (typeof data?.detail === "string") {
    return data.detail;
  }

  if (typeof data?.message === "string") {
    return data.message;
  }

  if (Array.isArray(data?.detail)) {
    return data.detail
      .map((item) => item.msg || JSON.stringify(item))
      .join(", ");
  }

  return fallback;
}

function prettyLabel(value) {
  return String(value)
    .replace(/_/g, " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function formatValue(value) {
  if (value === null || value === undefined) {
    return "Not available";
  }

  if (typeof value === "boolean") {
    return value ? "Yes" : "No";
  }

  if (typeof value === "number") {
    return Number.isInteger(value)
      ? value.toLocaleString()
      : value.toLocaleString(undefined, {
          maximumFractionDigits: 2,
        });
  }

  if (typeof value === "string") {
    return value;
  }

  if (Array.isArray(value)) {
    return value.map(formatValue).join(", ");
  }

  return JSON.stringify(value, null, 2);
}

function isPercentageKey(key) {
  return /score|percentage|percent|accuracy|coverage/i.test(key);
}

function formatScore(value) {
  if (typeof value !== "number") {
    return value;
  }

  // Values between 0 and 1 are commonly represented as fractions.
  // Other values are displayed without conversion.
  if (value >= 0 && value <= 1) {
    return `${(value * 100).toFixed(1)}%`;
  }

  return `${value.toFixed(1)}%`;
}

function formatFileSize(bytes) {
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }

  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function getNestedValue(object, path) {
  return path.split(".").reduce((value, key) => value?.[key], object);
}

function firstAvailable(object, paths) {
  for (const path of paths) {
    const value = getNestedValue(object, path);

    if (value !== undefined && value !== null) {
      return value;
    }
  }

  return null;
}

function collectText(value, output = [], depth = 0, parentKey = "") {
  if (depth > 8 || value === null || value === undefined) return output;

  const ignoredKey =
    /(^|_)(id|status_code|latency|duration|token_count|prompt_tokens|completion_tokens|text_length|word_count|keyword_frequency|raw_response|debug|metadata)(_|$)/i;

  if (ignoredKey.test(parentKey)) return output;

  if (typeof value === "string") {
    const text = value.trim();
    if (
      text &&
      !/^(success|complete|completed|true|false|null|none)$/i.test(text) &&
      !/^[{[]/.test(text)
    ) output.push(text);
    return output;
  }

  if (Array.isArray(value)) {
    // Keep narrative recommendation arrays, but do not flatten every skill,
    // keyword, requirement, or evidence item into the numbered AI analysis.
    // Those are better displayed in their dedicated compact sections.
    const isNarrativeArray = /recommendation|suggestion|improvement|feedback|advice|action|next_step|tip/i.test(parentKey);
    if (!isNarrativeArray) return output;
    value.forEach((item) => {
      if (typeof item === "string") {
        const text = item.trim();
        if (text && text.length > 35) output.push(text);
      } else {
        collectText(item, output, depth + 1, parentKey);
      }
    });
    return output;
  }

  if (typeof value === "object") {
    Object.entries(value).forEach(([key, item]) => {
      if (ignoredKey.test(key)) return;
      if (
        typeof item === "string" &&
        /explanation|recommendation|suggestion|summary|feedback|advice|action|improvement|strength|gap|priority|next_step|skill|reason|analysis|tip/i.test(key)
      ) {
        collectText(item, output, depth + 1, key);
      } else if (typeof item === "object" && item !== null) {
        collectText(item, output, depth + 1, key);
      }
    });
  }

  return output;
}

function findArray(object, paths) {
  for (const path of paths) {
    const value = getNestedValue(object, path);
    if (Array.isArray(value)) return value;
  }
  return [];
}

function findScore(object, paths) {
  for (const path of paths) {
    const value = getNestedValue(object, path);
    if (typeof value === "number") return value >= 0 && value <= 1 ? value * 100 : value;
    if (typeof value === "string" && value.trim() && !Number.isNaN(Number(value.replace("%", "")))) {
      return Number(value.replace("%", ""));
    }
  }
  return null;
}

function isSectionPresent(value) {
  if (typeof value === "boolean") return value;
  if (typeof value === "string") return !["no", "false", "missing", "absent"].includes(value.trim().toLowerCase());
  return Boolean(value);
}

function CompactList({ title, items, emptyText, limit = Infinity }) {
  const list = Array.isArray(items) ? items : [];
  if (!list.length) return null;

  // Some APIs return keyword arrays as one concatenated string. Split known
  // technology/education terms into individual chips and remove duplicates.
  const isKeywordList = /keywords identified/i.test(title);
  const knownKeywords = [
    "object-oriented programming", "machine learning", "real-time", "react.js",
    "node.js", "express.js", "mongodb", "python", "javascript", "typescript",
    "software engineering", "technical", "quality", "prediction", "movie",
    "integrated m.tech", "integrated", "github", "data", "state board",
    "sql", "nosql", "rest apis", "api development", "deployment", "docker",
    "debugging", "problem solving", "learning", "fruit", "express", "react",
    "node", "git", "java", "html", "css", "vit"
  ].sort((a, b) => b.length - a.length);

  const normalizeKeywords = (value) => {
    const source = String(value || "").replace(/[_•|]+/g, " ").replace(/\s+/g, " ").trim();
    if (!source) return [];
    const found = [];
    let remaining = source.toLowerCase();
    for (const keyword of knownKeywords) {
      const escaped = keyword.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      const re = new RegExp(escaped, "i");
      if (re.test(remaining)) {
        found.push(keyword);
        remaining = remaining.replace(re, " ");
      }
    }
    // If the API already returned clean individual keywords, preserve them.
    if (!found.length) return [source];
    const leftovers = remaining.split(/\s+/).filter(word => word.length > 2 && !["and", "the", "with"].includes(word));
    return [...new Set([...found, ...leftovers])];
  };

  const labels = list.flatMap(item => {
    const label = typeof item === "string"
      ? item
      : item?.keyword || item?.name || item?.requirement || item?.skill || item?.section || "";
    return isKeywordList ? normalizeKeywords(label) : [String(label || "").trim()];
  }).filter(Boolean);
  const uniqueLabels = [...new Map(labels.map(label => [label.toLowerCase(), label])).values()];

  return (
    <div className={`compact-result-group${isKeywordList ? " keyword-result-group" : ""}`}>
      <h4>{title}</h4>
      <div className="tag-list">
        {uniqueLabels.slice(0, limit).map((label, index) => (
          <span className="result-tag" key={`${label}-${index}`}>{label}</span>
        ))}
      </div>
    </div>
  );
}

// --------------------------------------------------
// Reusable components
// --------------------------------------------------

function Brand() {
  return (
    <div className="brand">
      <div className="brand-mark">
        <FileCheck2 size={24} strokeWidth={2.2} />
      </div>

      <div className="brand-copy">
        <span className="brand-title">ResumeIQ</span>
        <span className="brand-subtitle">AI CAREER WORKSPACE</span>
      </div>
    </div>
  );
}

function SectionHeading({ eyebrow, title, description, action }) {
  return (
    <div className="section-heading">
      <div>
        {eyebrow && <p className="eyebrow">{eyebrow}</p>}
        <h1>{title}</h1>
        {description && <p className="section-description">{description}</p>}
      </div>

      {action && <div className="heading-action">{action}</div>}
    </div>
  );
}

function StatusBadge({ status }) {
  const healthy = status === "healthy";

  return (
    <span className={`status-badge ${healthy ? "status-success" : "status-error"}`}>
      <span className="status-dot" />
      {healthy ? "Backend connected" : "Backend unavailable"}
    </span>
  );
}

function StatCard({ icon: Icon, label, value, detail, color = "blue" }) {
  return (
    <div className="stat-card">
      <div className={`stat-icon ${color}`}>
        <Icon size={21} />
      </div>

      <div className="stat-content">
        <span className="stat-label">{label}</span>
        <strong className="stat-value">{value}</strong>
        <span className="stat-detail">{detail}</span>
      </div>
    </div>
  );
}

function EmptyState({ icon: Icon = FileText, title, description }) {
  return (
    <div className="empty-state">
      <div className="empty-icon">
        <Icon size={30} />
      </div>

      <h3>{title}</h3>
      <p>{description}</p>
    </div>
  );
}

function LoadingButton({ loading, children, icon: Icon, ...props }) {
  return (
    <button {...props} disabled={loading || props.disabled}>
      {loading ? (
        <LoaderCircle className="spin" size={18} />
      ) : Icon ? (
        <Icon size={18} />
      ) : null}

      {loading ? "Processing..." : children}
    </button>
  );
}

// --------------------------------------------------
// Generic result renderer
// --------------------------------------------------

function ResultValue({ value }) {
  if (value === null || value === undefined) {
    return <span className="muted">Not available</span>;
  }

  if (typeof value === "boolean") {
    return (
      <span className={`boolean-value ${value ? "positive" : "negative"}`}>
        {value ? "Yes" : "No"}
      </span>
    );
  }

  if (Array.isArray(value)) {
    if (value.length === 0) {
      return <span className="muted">None identified</span>;
    }

    const primitiveArray = value.every(
      (item) => typeof item !== "object" || item === null
    );

    if (primitiveArray) {
      return (
        <div className="tag-list">
          {value.map((item, index) => (
            <span className="result-tag" key={`${String(item)}-${index}`}>
              {formatValue(item)}
            </span>
          ))}
        </div>
      );
    }

    return (
      <div className="nested-list">
        {value.map((item, index) => (
          <div className="nested-item" key={index}>
            {typeof item === "object" && item !== null ? (
              <ResultObject data={item} />
            ) : (
              formatValue(item)
            )}
          </div>
        ))}
      </div>
    );
  }

  if (typeof value === "object") {
    return <ResultObject data={value} />;
  }

  return <span className="result-value">{String(value)}</span>;
}

function ResultObject({ data, depth = 0 }) {
  if (!data || typeof data !== "object" || depth > 5) {
    return <ResultValue value={data} />;
  }

  if (Array.isArray(data)) {
    return <ResultValue value={data} />;
  }

  const entries = Object.entries(data);

  if (!entries.length) {
    return <span className="muted">No data available.</span>;
  }

  return (
    <div className={`result-object depth-${Math.min(depth, 3)}`}>
      {entries.map(([key, value]) => {
        const normalizedKey = key.toLowerCase();

        const isLongText =
          typeof value === "string" && value.length > 180;

        const isNested =
          value !== null && typeof value === "object";

        return (
          <div className="result-field" key={key}>
            <div className="result-field-label">
              {prettyLabel(key)}
            </div>

            {isPercentageKey(key) && typeof value === "number" ? (
              <div className="percentage-value">
                <strong>{formatScore(value)}</strong>
                <div className="percentage-track">
                  <div
                    className="percentage-fill"
                    style={{
                      width: `${Math.max(
                        0,
                        Math.min(
                          100,
                          value >= 0 && value <= 1 ? value * 100 : value
                        )
                      )}%`,
                    }}
                  />
                </div>
              </div>
            ) : isLongText ? (
              <p className="long-result-text">{value}</p>
            ) : isNested ? (
              <ResultObject data={value} depth={depth + 1} />
            ) : (
              <ResultValue value={value} />
            )}

            {normalizedKey === "status" && typeof value === "string" && (
              <span className="status-description">{value}</span>
            )}
          </div>
        );
      })}
    </div>
  );
}


function RecommendedImprovementCard({ item, index }) {
  if (typeof item === "string") {
    return (
      <div className="improvement-card" key={`${item}-${index}`}>
        <p className="improvement-message">{item}</p>
      </div>
    );
  }

  if (!item || typeof item !== "object") return null;

  const requirements = Array.isArray(item.requirements)
    ? item.requirements
    : Array.isArray(item.skills)
      ? item.skills
      : [];
  const isMissing = item.type === "missing_requirements";
  const title = isMissing
    ? "Missing Requirements"
    : item.type === "strengthen_evidence"
      ? "Strengthen Your Evidence"
      : item.title || item.type
        ? prettyLabel(item.title || item.type)
        : "Recommendation";

  const message = item.message || item.description || item.recommendation || "";

  return (
    <div className="improvement-card" key={`${item.type || title}-${index}`}>
      <div className="improvement-card-header">
        <h4>{title}</h4>
        {requirements.length > 0 && (
          <span className="improvement-count">
            {requirements.length} {requirements.length === 1 ? "skill" : "skills"}
          </span>
        )}
      </div>

      {message && <p className="improvement-message">{message}</p>}

      {requirements.length > 0 && (
        <p className="improvement-requirements-inline">
          <span className="improvement-requirements-label">Focus areas: </span>
          {requirements
            .map((requirement) =>
              typeof requirement === "string"
                ? requirement
                : requirement?.name ||
                  requirement?.skill ||
                  requirement?.requirement ||
                  requirement?.title ||
                  ""
            )
            .filter(Boolean)
            .join(", ")}
        </p>
      )}

      {requirements.length === 0 && !message && (
        <div className="improvement-skills-grid">
          {Object.entries(item)
            .filter(([key]) => !["type", "id", "status"].includes(key.toLowerCase()))
            .map(([key, value]) => (
              <div className="improvement-fallback" key={key}>
                <strong>{prettyLabel(key)}</strong>
                <ResultValue value={value} />
              </div>
            ))}
        </div>
      )}
    </div>
  );
}

// --------------------------------------------------
// Main application
// --------------------------------------------------

export default function App() {
  const [activePage, setActivePage] = useState("home");

  const [backendStatus, setBackendStatus] = useState("checking");
  const [backendMessage, setBackendMessage] = useState("");

  const [resumeFile, setResumeFile] = useState(null);
  const [resumeText, setResumeText] = useState("");

  const [jobDescription, setJobDescription] = useState("");

  const [extractionResult, setExtractionResult] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [matchingResult, setMatchingResult] = useState(null);
  const [llmResult, setLlmResult] = useState(null);

  const [loadingAction, setLoadingAction] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [successMessage, setSuccessMessage] = useState("");

  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  const [copied, setCopied] = useState(false);

  // ------------------------------------------------
  // Backend health check
  // ------------------------------------------------

  const checkBackend = useCallback(async () => {
    setBackendStatus("checking");
    setBackendMessage("");

    try {
      const response = await fetch(`${API_BASE_URL}/health`);

      const data = await response.json();

      if (!response.ok) {
        throw new Error(getErrorMessage(data, "Health check failed."));
      }

      setBackendStatus(
        data.status === "healthy" ? "healthy" : "unavailable"
      );

      setBackendMessage(data.message || "API is running.");
    } catch {
      setBackendStatus("unavailable");

      setBackendMessage(
        `Cannot connect to ${API_BASE_URL}. Check that the backend is running.`
      );
    }
  }, []);

  useEffect(() => {
    checkBackend();
  }, [checkBackend]);

  // ------------------------------------------------
  // Common helpers
  // ------------------------------------------------

  function clearNotices() {
    setErrorMessage("");
    setSuccessMessage("");
  }

  function navigate(page) {
    setActivePage(page);
    setSidebarOpen(false);
    clearNotices();
  }

  function handleFile(file) {
    clearNotices();

    if (!file) {
      return;
    }

    const extension = file.name.split(".").pop()?.toLowerCase();

    if (!["pdf", "docx"].includes(extension)) {
      setErrorMessage("Please upload a PDF or DOCX resume.");
      return;
    }

    if (file.size === 0) {
      setErrorMessage("The selected file is empty.");
      return;
    }

    if (file.size > MAX_FILE_SIZE) {
      setErrorMessage("The resume must not exceed 5 MB.");
      return;
    }

    setResumeFile(file);

    setResumeText("");
    setExtractionResult(null);
    setAnalysisResult(null);
    setMatchingResult(null);
    setLlmResult(null);

    setSuccessMessage("Resume selected successfully.");
  }

  function handleFileInput(event) {
    handleFile(event.target.files?.[0]);
    event.target.value = "";
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragActive(false);

    handleFile(event.dataTransfer.files?.[0]);
  }

  function removeFile() {
    setResumeFile(null);
    setResumeText("");
    setExtractionResult(null);
    setAnalysisResult(null);
    setMatchingResult(null);
    setLlmResult(null);

    clearNotices();
  }

  async function requestJson(endpoint, options = {}) {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, options);

    const contentType = response.headers.get("content-type") || "";

    let data;

    if (contentType.includes("application/json")) {
      data = await response.json();
    } else {
      data = await response.text();
    }

    if (!response.ok) {
      throw new Error(
        getErrorMessage(
          typeof data === "object" ? data : null,
          typeof data === "string" && data
            ? data
            : `Request failed with status ${response.status}.`
        )
      );
    }

    return data;
  }

  function createResumeForm() {
    if (!resumeFile) {
      throw new Error("Please upload a resume first.");
    }

    const formData = new FormData();

    // Must match the UploadFile parameter in FastAPI.
    formData.append("resume", resumeFile);

    return formData;
  }

  function createJobForm() {
    if (!jobDescription.trim()) {
      throw new Error("Please enter a job description.");
    }

    if (jobDescription.length > 50000) {
      throw new Error(
        "The job description must not exceed 50,000 characters."
      );
    }

    const formData = createResumeForm();

    // Must match the Form parameter in FastAPI.
    formData.append("job_description", jobDescription.trim());

    return formData;
  }

  async function runAction(action, callback) {
    clearNotices();
    setLoadingAction(action);

    try {
      await callback();
    } catch (error) {
      setErrorMessage(
        error instanceof Error
          ? error.message
          : "Something went wrong. Please try again."
      );
    } finally {
      setLoadingAction("");
    }
  }

  // ------------------------------------------------
  // API operations
  // ------------------------------------------------

  async function extractResume() {
    await runAction("extract", async () => {
      const data = await requestJson("/resume/extract", {
        method: "POST",
        body: createResumeForm(),
      });

      setExtractionResult(data);

      if (typeof data.resume_text === "string") {
        setResumeText(data.resume_text);
      }

      setSuccessMessage("Resume text extracted successfully.");
    });
  }

  async function analyzeResume() {
    await runAction("analyze", async () => {
      const data = await requestJson("/resume/analyze", {
        method: "POST",
        body: createResumeForm(),
      });

      setAnalysisResult(data);

      setSuccessMessage("Resume analysis completed.");

      navigate("analyzer");
    });
  }

  async function matchResume() {
    await runAction("match", async () => {
      const data = await requestJson("/resume/match", {
        method: "POST",
        body: createJobForm(),
      });

      setMatchingResult(data);

      setSuccessMessage("Resume and job description matched successfully.");

      navigate("matcher");
    });
  }

  async function getRecommendations() {
    await runAction("llm", async () => {
      const data = await requestJson("/resume/analyze-llm", {
        method: "POST",
        body: createJobForm(),
      });

      setLlmResult(data);

      // This endpoint also returns matching results.
      if (data.matching_result) {
        setMatchingResult({
          filename: data.filename,
          matching_result: data.matching_result,
        });
      }

      setSuccessMessage("AI analysis completed successfully.");

      navigate("recommendations");
    });
  }

  async function copyResumeText() {
    if (!resumeText) {
      setErrorMessage("Extract your resume text before copying it.");
      return;
    }

    try {
      await navigator.clipboard.writeText(resumeText);

      setCopied(true);

      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setErrorMessage(
        "Clipboard access failed. Please select and copy the text manually."
      );
    }
  }

  // ------------------------------------------------
  // Derived values
  // ------------------------------------------------

  const matchingPayload = matchingResult?.matching_result;

  const matchScore = useMemo(() => {
    if (!matchingPayload || typeof matchingPayload !== "object") {
      return null;
    }

    const value = firstAvailable(matchingPayload, [
      "match_score",
      "overall_match_score",
      "score",
      "match_percentage",
      "overall_score",
    ]);

    if (typeof value !== "number") {
      return null;
    }

    return value >= 0 && value <= 1
      ? Math.round(value * 100)
      : Math.round(value);
  }, [matchingPayload]);

  const recommendationTexts = useMemo(() => {
    if (!llmResult) {
      return [];
    }

    // The API may return explanatory boilerplate alongside useful findings.
    // Remove generic wrapper messages so they are not presented as separate
    // recommendations to the user.
    const boilerplatePatterns = [
      /^these job requirements were not found in your resume\.?/i,
      /^related evidence was found for these general requirements\.?/i,
      /^results generated from the response returned by your api\.?$/i,
      /^no recommendations returned\.?$/i,
      /^the following (skills|requirements|keywords) (were|are) (identified|found)/i,
      /^(missing requirements|strengthen your evidence|recommended improvements)\s*:?$/i,
      /^add specific supporting details\.?$/i,
      /^if you have relevant experience, add specific supporting details/i,
    ];

    const cleaned = collectText(llmResult.llm_analysis ?? llmResult)
      .map((item) => item
        .trim()
        .replace(/^\s*(?:\d+[.)]|[-*•])\s*/, "")
        .replace(/\s+/g, " "))
      .filter((item) => item.length > 0)
      .filter((item) => !boilerplatePatterns.some((pattern) => pattern.test(item)));

    // Keep the candidate overview first, then show only distinct, useful
    // recommendations. Generic matcher explanations and repeated strings are
    // intentionally excluded from this presentation layer.
    return [...new Set(cleaned)];
  }, [llmResult]);

  const currentPage = NAVIGATION.find((item) => item.id === activePage);

  // ------------------------------------------------
  // Sidebar
  // ------------------------------------------------

  function renderSidebar() {
    return (
      <>
        <div
          className={`sidebar-backdrop ${sidebarOpen ? "visible" : ""}`}
          onClick={() => setSidebarOpen(false)}
        />

        <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>
          <div className="sidebar-brand">
            <Brand />

            <button
              className="icon-button sidebar-close"
              onClick={() => setSidebarOpen(false)}
              aria-label="Close navigation"
            >
              <X size={19} />
            </button>
          </div>

          <div className="workspace-label">WORKSPACE</div>

          <nav className="sidebar-navigation">
            {NAVIGATION.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                className={`nav-item ${
                  activePage === id ? "nav-item-active" : ""
                }`}
                onClick={() => navigate(id)}
              >
                <Icon size={19} strokeWidth={1.9} />
                <span>{label}</span>

                {activePage === id && (
                  <ChevronRight className="nav-chevron" size={16} />
                )}
              </button>
            ))}
          </nav>

          <div className="sidebar-divider" />

          <div className="sidebar-info">
            <div className="sidebar-info-icon">
              <ShieldCheck size={20} />
            </div>

            <strong>Your data, your career</strong>

            <p>
              Analyze resumes and compare skills with job requirements.
            </p>
          </div>

          <div className="sidebar-bottom">
            <div className="sidebar-status-row">
              <span
                className={`status-dot ${
                  backendStatus === "healthy"
                    ? "dot-green"
                    : backendStatus === "checking"
                    ? "dot-yellow"
                    : "dot-red"
                }`}
              />

              <span>
                {backendStatus === "healthy"
                  ? "API connected"
                  : backendStatus === "checking"
                  ? "Checking API..."
                  : "API disconnected"}
              </span>
            </div>

            <span className="version-label">ResumeIQ · Version 1.0</span>
          </div>
        </aside>
      </>
    );
  }

  // ------------------------------------------------
  // Header
  // ------------------------------------------------

  function renderHeader() {
    return (
      <header className="topbar">
        <div className="topbar-left">
          <button
            className="icon-button mobile-menu"
            onClick={() => setSidebarOpen(true)}
            aria-label="Open navigation"
          >
            <Menu size={22} />
          </button>

          <div className="breadcrumb">
            <span>Workspace</span>
            <ChevronRight size={15} />
            <strong>{currentPage?.label || "Dashboard"}</strong>
          </div>
        </div>

        <div className="topbar-actions">
          <StatusBadge
            status={backendStatus === "healthy" ? "healthy" : "unhealthy"}
          />

          <button
            className="button button-outline button-small"
            onClick={checkBackend}
            disabled={backendStatus === "checking"}
          >
            {backendStatus === "checking" ? (
              <LoaderCircle className="spin" size={16} />
            ) : (
              <HeartPulse size={16} />
            )}

            <span>Health Check</span>
          </button>
        </div>
      </header>
    );
  }

  // ------------------------------------------------
  // Notices
  // ------------------------------------------------

  function renderNotices() {
    return (
      <>
        {errorMessage && (
          <div className="notice notice-error" role="alert">
            <CircleAlert size={19} />

            <div>
              <strong>Something went wrong</strong>
              <p>{errorMessage}</p>
            </div>

            <button
              className="notice-close"
              onClick={() => setErrorMessage("")}
              aria-label="Dismiss error"
            >
              <X size={17} />
            </button>
          </div>
        )}

        {successMessage && (
          <div className="notice notice-success" role="status">
            <CheckCircle2 size={19} />

            <p>{successMessage}</p>

            <button
              className="notice-close"
              onClick={() => setSuccessMessage("")}
              aria-label="Dismiss message"
            >
              <X size={17} />
            </button>
          </div>
        )}
      </>
    );
  }

  // ------------------------------------------------
  // File uploader
  // ------------------------------------------------

  function renderUploader() {
    return (
      <div className="upload-card">
        <div className="card-heading">
          <div className="card-heading-icon">
            <CloudUpload size={22} />
          </div>

          <div>
            <h3>Upload your resume</h3>
            <p>Choose a PDF or DOCX document to get started.</p>
          </div>
        </div>

        <label
          className={`dropzone ${dragActive ? "dropzone-active" : ""}`}
          onDragOver={(event) => {
            event.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={handleDrop}
        >
          <input
            type="file"
            accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            onChange={handleFileInput}
          />

          <div className="dropzone-icon">
            <Upload size={27} />
          </div>

          <strong>
            {dragActive ? "Drop your resume here" : "Drag and drop your resume"}
          </strong>

          <span>or click to browse files</span>

          <div className="upload-format">
            <FileText size={14} />
            PDF or DOCX · Maximum 5 MB
          </div>
        </label>

        {resumeFile && (
          <div className="selected-file">
            <div className="selected-file-icon">
              <FileText size={21} />
            </div>

            <div className="selected-file-info">
              <strong title={resumeFile.name}>{resumeFile.name}</strong>
              <span>{formatFileSize(resumeFile.size)}</span>
            </div>

            <span className="file-ready">
              <Check size={14} />
              Ready
            </span>

            <button
              className="icon-button remove-file"
              onClick={removeFile}
              aria-label="Remove selected resume"
            >
              <Trash2 size={17} />
            </button>
          </div>
        )}

        <div className="upload-actions">
          <LoadingButton
            className="button button-primary"
            icon={Upload}
            loading={loadingAction === "extract"}
            disabled={!resumeFile || loadingAction !== ""}
            onClick={extractResume}
          >
            Extract Resume Text
          </LoadingButton>

          <LoadingButton
            className="button button-secondary"
            icon={BarChart3}
            loading={loadingAction === "analyze"}
            disabled={!resumeFile || loadingAction !== ""}
            onClick={analyzeResume}
          >
            Analyze Resume
          </LoadingButton>
        </div>
      </div>
    );
  }

  // ------------------------------------------------
  // Resume extraction
  // ------------------------------------------------

  function renderExtractedText() {
    const characterCount = resumeText.length;

    const wordCount = resumeText.trim()
      ? resumeText.trim().split(/\s+/).length
      : 0;

    return (
      <div className="panel-card extraction-card">
        <div className="panel-heading">
          <div className="panel-heading-left">
            <div className="panel-icon blue-icon">
              <FileText size={20} />
            </div>

            <div>
              <h3>Extracted resume text</h3>
              <p>
                {extractionResult
                  ? extractionResult.filename
                  : "Extracted content will appear here."}
              </p>
            </div>
          </div>

          <button
            className="button button-outline button-small"
            onClick={copyResumeText}
            disabled={!resumeText}
          >
            {copied ? <Check size={15} /> : <Clipboard size={15} />}
            {copied ? "Copied" : "Copy text"}
          </button>
        </div>

        {resumeText ? (
          <>
            <textarea
              className="resume-textarea"
              value={resumeText}
              onChange={(event) => setResumeText(event.target.value)}
              spellCheck="false"
              aria-label="Extracted resume text"
            />

            <div className="text-statistics">
              <span>
                <FileText size={14} />
                {characterCount.toLocaleString()} characters
              </span>

              <span>
                <Activity size={14} />
                {wordCount.toLocaleString()} words
              </span>
            </div>
          </>
        ) : (
          <EmptyState
            icon={FileText}
            title="No resume text yet"
            description="Upload your resume and select Extract Resume Text to see the extracted content here."
          />
        )}
      </div>
    );
  }

  // ------------------------------------------------
  // Home dashboard
  // ------------------------------------------------

  function renderHome() {
    return (
      <>
        <section className="welcome-banner">
          <div className="welcome-content">
            <div className="welcome-badge">
              <Sparkles size={14} />
              AI-POWERED CAREER TOOLS
            </div>

            <h1>
              Your next career move
              <br />
              starts <span>with better insights.</span>
            </h1>

            <p>
              Understand your resume, identify skill gaps, and discover
              how well your experience matches a job.
            </p>

            <div className="welcome-actions">
              <button
                className="button button-white"
                onClick={() => navigate("analyzer")}
              >
                Analyze my resume
                <ArrowRight size={17} />
              </button>

              <button
                className="button button-glass"
                onClick={() => navigate("matcher")}
              >
                Match a job
              </button>
            </div>
          </div>

          <div className="welcome-illustration">
            <div className="illustration-glow" />

            <div className="illustration-document">
              <div className="illustration-document-top">
                <div className="illustration-avatar">
                  <FileText size={28} />
                </div>

                <div>
                  <span className="illustration-line long-line" />
                  <span className="illustration-line short-line" />
                </div>
              </div>

              <div className="illustration-line full-line" />
              <div className="illustration-line medium-line" />

              <div className="illustration-skill-row">
                <span />
                <span />
                <span />
              </div>

              <div className="illustration-chart">
                <div className="chart-bar bar-one" />
                <div className="chart-bar bar-two" />
                <div className="chart-bar bar-three" />
                <div className="chart-bar bar-four" />
                <div className="chart-bar bar-five" />
              </div>
            </div>

            <div className="floating-score">
              <div className="floating-score-icon">
                <Target size={20} />
              </div>

              <div>
                <span>Job compatibility</span>
                <strong>Find your fit</strong>
              </div>
            </div>

            <div className="floating-sparkle">
              <Sparkles size={20} />
            </div>
          </div>
        </section>

        <div className="section-title-row">
          <div>
            <h2>Workspace overview</h2>
            <p>Everything you need to prepare for your next opportunity.</p>
          </div>
        </div>

        <section className="stats-grid">
          <StatCard
            icon={FileCheck2}
            label="Resume status"
            value={resumeFile ? "Uploaded" : "Not uploaded"}
            detail={resumeFile ? resumeFile.name : "Ready for your resume"}
            color="blue"
          />

          <StatCard
            icon={BarChart3}
            label="Resume analysis"
            value={analysisResult ? "Completed" : "Ready to start"}
            detail={
              analysisResult
                ? "Results available"
                : "Review your resume content"
            }
            color="purple"
          />

          <StatCard
            icon={Target}
            label="Job matching"
            value={
              matchScore !== null ? `${matchScore}%` : "Not calculated"
            }
            detail={
              matchScore !== null
                ? "Latest match score"
                : "Compare against a job"
            }
            color="green"
          />

          <StatCard
            icon={Sparkles}
            label="AI recommendations"
            value={llmResult ? "Available" : "Ready to explore"}
            detail={
              llmResult
                ? "AI analysis completed"
                : "Personalized job insights"
            }
            color="orange"
          />
        </section>

        <div className="section-title-row section-title-spaced">
          <div>
            <h2>Explore your tools</h2>
            <p>Choose a task to get started.</p>
          </div>
        </div>

        <section className="feature-grid">
          <button
            className="feature-card"
            onClick={() => navigate("analyzer")}
          >
            <div className="feature-top">
              <div className="feature-icon blue-icon">
                <FileText size={22} />
              </div>

              <ArrowUpRight size={19} className="feature-arrow" />
            </div>

            <h3>Resume Analyzer</h3>

            <p>
              Extract your resume and review its structure and content.
            </p>

            <span className="feature-link">
              Open analyzer <ArrowRight size={15} />
            </span>
          </button>

          <button
            className="feature-card"
            onClick={() => navigate("matcher")}
          >
            <div className="feature-top">
              <div className="feature-icon green-icon">
                <Target size={22} />
              </div>

              <ArrowUpRight size={19} className="feature-arrow" />
            </div>

            <h3>Job Matcher</h3>

            <p>
              Compare your resume against a specific job description.
            </p>

            <span className="feature-link green-link">
              Find your match <ArrowRight size={15} />
            </span>
          </button>

          <button
            className="feature-card"
            onClick={() => navigate("recommendations")}
          >
            <div className="feature-top">
              <div className="feature-icon orange-icon">
                <Lightbulb size={22} />
              </div>

              <ArrowUpRight size={19} className="feature-arrow" />
            </div>

            <h3>AI Recommendations</h3>

            <p>
              Get AI-generated explanations and practical suggestions.
            </p>

            <span className="feature-link orange-link">
              Get recommendations <ArrowRight size={15} />
            </span>
          </button>
        </section>

        <section className="how-it-works">
          <div className="how-heading">
            <div className="how-icon">
              <Info size={19} />
            </div>

            <div>
              <h3>How it works</h3>
              <p>A simple workflow from resume to actionable insights.</p>
            </div>
          </div>

          <div className="workflow-steps">
            {[
              {
                icon: Upload,
                title: "Upload",
                text: "Choose a PDF or DOCX resume.",
              },
              {
                icon: FileCheck2,
                title: "Analyze",
                text: "Review resume content and structure.",
              },
              {
                icon: BriefcaseBusiness,
                title: "Match",
                text: "Compare with a job description.",
              },
              {
                icon: WandSparkles,
                title: "Improve",
                text: "Explore AI-powered recommendations.",
              },
            ].map(({ icon: Icon, title, text }, index) => (
              <div className="workflow-step" key={title}>
                <div className="workflow-step-number">{index + 1}</div>

                <div className="workflow-step-icon">
                  <Icon size={19} />
                </div>

                <strong>{title}</strong>
                <p>{text}</p>
              </div>
            ))}
          </div>
        </section>
      </>
    );
  }

  // ------------------------------------------------
  // Resume analyzer page
  // ------------------------------------------------

  function renderAnalyzer() {
    return (
      <>
        <SectionHeading
          eyebrow="RESUME INTELLIGENCE"
          title="Resume Analyzer"
          description="Extract your resume text and review the analysis returned by your backend."
          action={
            resumeFile && (
              <span className="file-count-badge">
                <CheckCircle2 size={15} />
                Resume selected
              </span>
            )
          }
        />

        <section className="analyzer-grid">
          {renderUploader()}
          {renderExtractedText()}
        </section>

        <section className="panel-card results-panel">
          <div className="panel-heading">
            <div className="panel-heading-left">
              <div className="panel-icon purple-icon">
                <BarChart3 size={20} />
              </div>

              <div>
                <h3>Resume analysis results</h3>
                <p>
                  The analysis is based on the response from your API.
                </p>
              </div>
            </div>

            <button
              className="button button-outline button-small"
              onClick={analyzeResume}
              disabled={!resumeFile || loadingAction !== ""}
            >
              <RefreshCw size={15} />
              Run analysis
            </button>
          </div>

          {analysisResult ? (() => {
            const result = analysisResult.analysis ?? analysisResult;
            const score = findScore(result, [
              "score", "resume_score", "overall_score", "completeness_score",
              "resume_completeness_score", "summary.score"
            ]);
            const missingSections = findArray(result, ["missing_sections", "missingSections"]);
            const keywords = findArray(result, ["keywords", "keyword_analysis", "top_keywords", "identified_keywords"]);
            const strengths = findArray(result, ["strengths", "resume_strengths", "positive_points"]);
            const suggestions = findArray(result, ["recommendations", "suggestions", "improvement_suggestions", "areas_to_improve"]);
            const sections = firstAvailable(result, ["sections", "section_presence"]);
            const sectionEntries = sections && typeof sections === "object" && !Array.isArray(sections)
              ? Object.entries(sections).filter(([key]) =>
                  ["summary", "education", "experience", "skills", "projects", "certifications"].includes(key.toLowerCase())
                )
              : [];
            return (
              <div className="compact-analysis">
                <div className="compact-score-card">
                  <div>
                    <span className="compact-kicker">RESUME COMPLETENESS</span>
                    <strong>{score !== null ? `${score.toFixed(1)}%` : "Analysis complete"}</strong>
                    <p>Preliminary completeness estimate, not an official ATS score.</p>
                  </div>
                  {score !== null && <div className="compact-score-track"><span style={{ width: `${Math.max(0, Math.min(100, score))}%` }} /></div>}
                </div>
                {sectionEntries.length > 0 && (
                  <div className="compact-result-group">
                    <h4>Key sections</h4>
                    <div className="section-status-list">
                      {sectionEntries.map(([name, present]) => (
                        <span className={`section-status ${isSectionPresent(present) ? "is-present" : "is-missing"}`} key={name}>
                          {isSectionPresent(present) ? <CheckCircle2 size={14} /> : <CircleAlert size={14} />}
                          {prettyLabel(name)}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                <CompactList title="Keywords identified in your resume" items={keywords} />
                <CompactList title="Resume strengths" items={strengths} />
                <CompactList title="Suggested improvements" items={suggestions} />
                {missingSections.length > 0 && (
                  <div className="compact-note">
                    <strong>Sections to consider adding:</strong>{" "}
                    {missingSections.map(item => typeof item === "string" ? item : item?.section || item?.name || item?.title || "").filter(Boolean).join(", ")}
                  </div>
                )}
                <div className="compact-detail-grid">
                  {[
                    ["Summary", ["summary", "resume_summary", "analysis_summary", "overview"]],
                    ["Feedback", ["feedback", "analysis_feedback", "explanation", "assessment"]],
                    ["Content to improve", ["areas_for_improvement", "improvements", "content_suggestions"]],
                  ].map(([title, paths]) => {
                    const value = firstAvailable(result, paths);
                    if (value === null || value === undefined || value === "") return null;
                    if (Array.isArray(value) && value.length === 0) return null;
                    return (
                      <div className="compact-detail-card" key={title}>
                        <h4>{title}</h4>
                        {Array.isArray(value) ? (
                          <ul>{value.map((item, index) => (
                            <li key={index}>{typeof item === "string" ? item : item?.recommendation || item?.suggestion || item?.description || item?.name || JSON.stringify(item)}</li>
                          ))}</ul>
                        ) : typeof value === "object" ? (
                          <ResultObject data={value} />
                        ) : (
                          <p>{String(value)}</p>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })() : (
            <EmptyState
              icon={BarChart3}
              title="Analysis not run yet"
              description="Upload a resume and select Analyze Resume to review the detailed results returned by your backend."
            />
          )}
        </section>
      </>
    );
  }

  // ------------------------------------------------
  // Job matcher page
  // ------------------------------------------------

  function renderMatcher() {
    return (
      <>
        <SectionHeading
          eyebrow="CAREER COMPATIBILITY"
          title="Job Matcher"
          description="Compare your resume against a job description to understand how your skills align with the role."
        />

        <section className="matcher-grid matcher-grid-stacked-results">
          <div className="matcher-top-grid">
            <div className="panel-card matcher-input-card">
            <div className="panel-heading">
              <div className="panel-heading-left">
                <div className="panel-icon blue-icon">
                  <FileText size={20} />
                </div>

                <div>
                  <h3>Your resume</h3>
                  <p>Choose the resume you want to evaluate.</p>
                </div>
              </div>
            </div>

            {resumeFile ? (
              <div className="matcher-file">
                <div className="selected-file-icon">
                  <FileText size={22} />
                </div>

                <div className="selected-file-info">
                  <strong>{resumeFile.name}</strong>
                  <span>{formatFileSize(resumeFile.size)}</span>
                </div>

                <CheckCircle2 className="success-text" size={19} />
              </div>
            ) : (
              <label className="compact-upload">
                <input
                  id="replace-resume"
                  type="file"
                  accept=".pdf,.docx,application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                  onChange={handleFileInput}
                />

                <CloudUpload size={23} />

                <strong>Upload your resume</strong>

                <span>PDF or DOCX · Maximum 5 MB</span>
              </label>
            )}

            {resumeFile && (
              <button
                className="text-button"
                onClick={() => document.getElementById("replace-resume")?.click()}
              >
                Replace resume
              </button>
            )}
            </div>
            <div className="panel-card matcher-description-card">
              <div className="panel-heading">
                <div className="panel-heading-left">
                  <div className="panel-icon blue-icon">
                    <FileText size={20} />
                  </div>
                  <div>
                    <h3>Job description</h3>
                    <p>Paste the role requirements to compare against your resume.</p>
                  </div>
                </div>
              </div>
              

            <label className="form-label" htmlFor="job-description">
              Job description
              <span className="required-mark">*</span>
            </label>

            <p className="form-help">
              Paste the full job description, including required skills,
              responsibilities, and qualifications.
            </p>

            <textarea
              id="job-description"
              className="job-description-input"
              placeholder={`Example:

We are looking for a Full Stack Developer.

Requirements:
- JavaScript and React
- Node.js and Express.js
- REST API development
- MongoDB
- Git and version control

Responsibilities:
- Build and maintain web applications
- Develop backend APIs
- Collaborate with the engineering team`}
              value={jobDescription}
              onChange={(event) => setJobDescription(event.target.value)}
              maxLength={50000}
            />

            <div className="textarea-footer">
              <span>
                {jobDescription.length.toLocaleString()} / 50,000 characters
              </span>

              <span
                className={
                  jobDescription.length > 48000 ? "warning-text" : ""
                }
              >
                {jobDescription.trim() ? "Ready to analyze" : "Required"}
              </span>
            </div>

            <div className="matcher-actions">
              <LoadingButton
                className="button button-primary button-full"
                icon={Target}
                loading={loadingAction === "match"}
                disabled={
                  !resumeFile ||
                  !jobDescription.trim() ||
                  loadingAction !== ""
                }
                onClick={matchResume}
              >
                Match Resume to Job
              </LoadingButton>

              <p className="privacy-note">
                <ShieldCheck size={15} />
                The match score is a heuristic, not a hiring prediction.
              </p>
            </div>
            </div>
          </div>
          <div className="matcher-results-full">
            <div className="panel-card matcher-results-card">
            <div className="panel-heading">
              <div className="panel-heading-left">
                <div className="panel-icon green-icon">
                  <Target size={20} />
                </div>

                <div>
                  <h3>Match results</h3>
                  <p>Review the returned matching data.</p>
                </div>
              </div>
            </div>

            {matchingResult ? (
              <>
                {matchScore !== null && (
                  <div className="score-overview">
                    <div
                      className="score-ring"
                      style={{
                        "--score": `${Math.max(
                          0,
                          Math.min(100, matchScore)
                        )}%`,
                      }}
                    >
                      <div className="score-ring-inner">
                        <strong>{matchScore}%</strong>
                        <span>Match score</span>
                      </div>
                    </div>

                    <div className="score-description">
                      <div className="score-status">
                        <CheckCircle2 size={17} />
                        Analysis completed
                      </div>

                      <p>
                        This score represents the matching result returned
                        by your backend. Review individual requirements
                        before drawing conclusions.
                      </p>
                    </div>
                  </div>
                )}

                <div className="result-divider" />

                <div className="matching-result-content compact-analysis">
                  <CompactList
                    title="Matched requirements"
                    items={findArray(matchingPayload, ["matched_requirements", "matched_skills", "matched_keywords", "matched"])}
                  />
                  <CompactList
                    title="Missing or unverified requirements"
                    items={findArray(matchingPayload, ["missing_requirements", "missing_skills", "unmatched_requirements", "missing_keywords", "unmatched"])}
                  />
                  {[
                    ["Match explanation", ["explanation", "match_explanation", "summary", "analysis_summary", "reasoning"]],
                    ["Relevant experience", ["relevant_experience", "experience_match", "matched_experience"]],
                  ].map(([title, paths]) => {
                    const value = firstAvailable(matchingPayload, paths);
                    if (value === null || value === undefined || value === "") return null;
                    return (
                      <div className="compact-detail-card" key={title}>
                        <h4>{title}</h4>
                        {title === "Recommended improvements" && Array.isArray(value) ? (
                          <div className="improvement-list">
                            {value
                              .filter((item) => {
                                if (!item || typeof item !== "object") return true;
                                const itemType = String(item.type || "").toLowerCase().replace(/[\s-]+/g, "_");
                                const itemTitle = String(item.title || "").toLowerCase().trim();
                                return itemType !== "missing_requirements" && itemTitle !== "missing requirements";
                              })
                              .map((item, index) => (
                                <RecommendedImprovementCard item={item} index={index} key={index} />
                              ))}
                          </div>
                        ) : Array.isArray(value) ? (
                          <ul>{value.map((item, index) => (
                            <li key={index}>{typeof item === "string" ? item : item?.description || item?.recommendation || item?.requirement || item?.name || JSON.stringify(item)}</li>
                          ))}</ul>
                        ) : typeof value === "object" ? <ResultObject data={value} /> : <p>{String(value)}</p>}
                      </div>
                    );
                  })}
                  <p className="compact-footnote">
                    This is an indicative match based on the supplied resume and job description, not a hiring prediction.
                  </p>
                </div>

                <button
                  className="button button-secondary button-full recommendation-cta"
                  onClick={() => navigate("recommendations")}
                >
                  View AI recommendations
                  <ArrowRight size={17} />
                </button>
              </>
            ) : (
              <EmptyState
                icon={Target}
                title="Ready to find your match?"
                description="Upload a resume, enter a job description, and run the matching analysis to view the results here."
              />
            )}
          </div>
          </div>
        </section>
      </>
    );
  }

  // ------------------------------------------------
  // AI recommendations page
  // ------------------------------------------------

  function renderRecommendations() {
    return (
      <>
        <SectionHeading
          eyebrow="AI CAREER ASSISTANT"
          title="AI Recommendations"
          description="Generate an explanation of your resume-to-job match and view recommendations returned by your Groq-powered backend."
        />

        <section className="recommendation-hero">
          <div className="recommendation-hero-icon">
            <Sparkles size={28} />
          </div>

          <div className="recommendation-hero-content">
            <span className="eyebrow">PERSONALIZED INSIGHTS</span>

            <h2>Turn your match results into next steps.</h2>

            <p>
              Provide a resume and job description to generate
              AI-powered explanations and recommendations.
            </p>

            <button
              className="button button-white"
              onClick={() => navigate("matcher")}
            >
              {resumeFile && jobDescription
                ? "Review job description"
                : "Prepare job match"}
              <ArrowRight size={17} />
            </button>
          </div>

          <div className="recommendation-decoration">
            <WandSparkles size={100} strokeWidth={1} />
          </div>
        </section>

        <section className="panel-card recommendation-form-card">
          <div className="panel-heading">
            <div className="panel-heading-left">
              <div className="panel-icon orange-icon">
                <Lightbulb size={20} />
              </div>

              <div>
                <h3>Generate AI analysis</h3>
                <p>
                  The AI endpoint also calculates the existing match result.
                </p>
              </div>
            </div>
          </div>

          <div className="recommendation-input-summary">
            <div className="input-summary-item">
              <div className="input-summary-icon">
                <FileText size={20} />
              </div>

              <div>
                <strong>Resume</strong>

                <span>
                  {resumeFile
                    ? resumeFile.name
                    : "No resume uploaded"}
                </span>
              </div>

              {!resumeFile && (
                <button
                  className="text-button"
                  onClick={() => navigate("analyzer")}
                >
                  Upload
                </button>
              )}
            </div>

            <div className="input-summary-item">
              <div className="input-summary-icon">
                <BriefcaseBusiness size={20} />
              </div>

              <div>
                <strong>Job description</strong>

                <span>
                  {jobDescription.trim()
                    ? `${jobDescription.trim().length} characters entered`
                    : "No job description entered"}
                </span>
              </div>

              {!jobDescription.trim() && (
                <button
                  className="text-button"
                  onClick={() => navigate("matcher")}
                >
                  Add job
                </button>
              )}
            </div>
          </div>

          <button
            className="button button-primary button-full"
            onClick={getRecommendations}
            disabled={
              !resumeFile ||
              !jobDescription.trim() ||
              loadingAction !== ""
            }
          >
            {loadingAction === "llm" ? (
              <LoaderCircle className="spin" size={18} />
            ) : (
              <Sparkles size={18} />
            )}

            {loadingAction === "llm"
              ? "Generating recommendations..."
              : "Generate AI Recommendations"}
          </button>
        </section>

        {llmResult ? (
          <section className="panel-card results-panel">
            <div className="panel-heading">
              <div className="panel-heading-left">
                <div className="panel-icon orange-icon">
                  <WandSparkles size={20} />
                </div>

                <div>
                  <h3>Your AI analysis</h3>
                  <p>
                    Results generated from the response returned by your API.
                  </p>
                </div>
              </div>
            </div>

            {recommendationTexts.length > 0 ? (
              <div className="ai-analysis-content">
                {recommendationTexts[0] && (
                  <div className="ai-analysis-summary">
                    <h4>Candidate overview</h4>
                    <p>{recommendationTexts[0]}</p>
                  </div>
                )}

                {recommendationTexts.length > 1 && (
                  <div className="ai-analysis-actions">
                    <h4>Recommended next steps</h4>
                    <ul>
                      {recommendationTexts.slice(1).map((text, index) => (
                        <li key={`${text.slice(0, 40)}-${index}`}>{text}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ) : (
              <EmptyState
                icon={Lightbulb}
                title="No recommendations returned"
                description="The AI service completed, but did not return readable recommendation text. Please try again."
              />
            )}
          </section>
        ) : (
          <section className="panel-card recommendation-empty-panel">
            <EmptyState
              icon={Sparkles}
              title="Your recommendations will appear here"
              description="Run AI analysis to view the explanation and recommendations generated by your backend."
            />
          </section>
        )}
      </>
    );
  }

  // ------------------------------------------------
  // Footer
  // ------------------------------------------------

  function renderFooter() {
    return (
      <footer className="app-footer">
        <div>
          <span className="footer-brand">ResumeIQ</span>
          <span className="footer-separator">·</span>
          <span>AI Resume Analyzer & Job Matcher</span>
        </div>

        <div className="footer-right">
          <span>
            <span
              className={`status-dot ${
                backendStatus === "healthy" ? "dot-green" : "dot-red"
              }`}
            />
            {backendStatus === "healthy" ? "API online" : "API status unavailable"}
          </span>

          <button
            className="footer-refresh"
            onClick={checkBackend}
            title="Refresh backend status"
          >
            <RefreshCw size={14} />
          </button>
        </div>
      </footer>
    );
  }

  // ------------------------------------------------
  // Render
  // ------------------------------------------------

  return (
    <div className="app-shell">
      {renderSidebar()}

      <div className="main-area">
        {renderHeader()}

        <main className="page-content">
          {renderNotices()}

          {activePage === "home" && renderHome()}
          {activePage === "analyzer" && renderAnalyzer()}
          {activePage === "matcher" && renderMatcher()}
          {activePage === "recommendations" && renderRecommendations()}

          {renderFooter()}
        </main>
      </div>
    </div>
  );
}