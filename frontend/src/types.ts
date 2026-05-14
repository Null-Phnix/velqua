/**
 * API response type definitions for Velqua.
 * These mirror the Pydantic models and serializers in the backend.
 */

export interface FactItem {
  id: string;
  content: string;
  type: string;
  confidence: number;
  confirmation_count: number;
  topic: string;
  category: string;
  emotion: string;
  sentiment_score: number;
  tags?: string[];
  timestamp?: string;
}

export interface FactListResponse {
  facts: FactItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface FactSearchResponse {
  query: string;
  results: FactItem[];
  count: number;
}

export interface FactTypeEntry {
  value: string;
  label: string;
}

export interface PendingFact {
  id: string;
  content: string;
  quality_score: number;
  detected_topic?: string;
  detected_category?: string;
  detected_emotion?: string;
  contradictions?: Array<{ content: string; confidence: number }>;
}

export interface PendingResponse {
  pending: PendingFact[];
  count: number;
}

export interface ProviderConfig {
  name: string;
  has_api_key?: boolean;
  base_url?: string;
  default_model?: string;
  models?: string[];
  enabled?: boolean;
}

export interface ProviderDef {
  name: string;
  label: string;
  desc: string;
  color: string;
  needsKey: boolean;
}

export interface Settings {
  budget: string;
  auto_learning: boolean;
  active_provider: string;
}

export interface ProvidersResponse {
  providers: ProviderConfig[];
}

export interface ConnectionTestResult {
  ok: boolean;
  models: string[];
  error?: string;
}

export interface LicenseStatusData {
  status: 'active' | 'trial' | 'expired' | 'invalid';
  is_active: boolean;
  is_trial: boolean;
  message?: string;
  customer_email?: string;
  product_name?: string;
}

export interface LicenseActivateResponse {
  success: boolean;
  message?: string;
}

export interface HealthData {
  facts_count: number;
  database_size_mb: number;
}

export interface FactStats {
  total: number;
  by_type: Record<string, number>;
  by_confidence: { high: number; medium: number; low: number };
}

export interface ProxyStatus {
  status: string;
  error?: string;
  backends?: Record<string, string>;
  memory_config?: { budget: string; max_tokens: number };
  vector_retrieval?: boolean;
  model_cached?: boolean | null;
  auto_learning?: { enabled: boolean; facts_learned: number; facts_pending: number };
}

export interface TimelineFact {
  id: string;
  content: string;
  type: string;
  confidence: number;
  topic: string;
  category: string;
  emotion: string;
  sentiment_score: number;
}

export interface TimelineData {
  dates: string[];
  groups: Record<string, TimelineFact[]>;
  total_facts: number;
  total_days: number;
}

export interface AnalyticsReport {
  error?: string;
  health: { healthy: number; aging: number; at_risk: number; forgotten: number };
  total_facts: number;
  total_episodes: number;
  memory_span_days: number;
  topic_diversity: number;
  top_topics: Array<{ topic: string; count: number }>;
  avg_fact_confidence: number;
  avg_episode_importance: number;
  emotional_balance: number;
  facts_by_type: Record<string, number>;
}

export interface QualityStats {
  avg_quality: number;
  total: number;
  distribution: Record<string, number>;
  common_issues: string[];
}

export interface QualityReport {
  error?: string;
  stats?: QualityStats;
}

export interface GraphStats {
  error?: string;
  total_links: number;
  by_type: Record<string, number>;
}

export interface EmotionalHistory {
  error?: string;
  episode_count?: number;
  dominant_valence?: string;
  avg_sentiment?: number;
  valence_distribution?: Record<string, number>;
}

export interface ContradictionPair {
  fact_a: { id: string; content: string };
  fact_b: { id: string; content: string };
  type: string;
  confidence: number;
}

export interface ContradictionsResponse {
  error?: string;
  count: number;
  contradictions: ContradictionPair[];
}

export interface PreviewInjectedFact {
  content: string;
  score: number;
  topic_boost: number;
}

export interface PreviewData {
  facts_injected: number;
  facts_available: number;
  tokens_used: number;
  token_budget: number;
  search_mode: string;
  query_category?: string;
  injected: PreviewInjectedFact[];
  skipped: unknown[];
  context: string;
}

export interface BackupItem {
  filename: string;
  size_mb: number;
}

export interface BackupListResponse {
  backups: BackupItem[];
}

export interface BackupCreateResponse {
  backup_path: string;
  size_mb: number;
}

export interface ImportHistoryEntry {
  batch_id: string;
  file_type: string;
  facts_stored: number;
  duplicates_skipped: number;
  timestamp: number;
  undone: boolean;
  fact_ids?: string[];
}

export interface ImportHistoryResponse {
  history: ImportHistoryEntry[];
}

export interface ExportResponse {
  count: number;
  [key: string]: unknown;
}

export interface CompactResponse {
  message: string;
  superseded: number;
}

export interface AgentInfo {
  id: string;
  name: string;
  is_active: boolean;
  last_seen_ago: number;
  current_task?: string;
}

export interface MeshAgentsResponse {
  agents: AgentInfo[];
}

export interface MeshMemoryEntry {
  agent_id: string;
  timestamp: number;
  content: string;
  tags?: string[];
}

export interface MeshMemoryResponse {
  entries: MeshMemoryEntry[];
}

export interface MeshNote {
  id: string;
  from_agent: string;
  to_agent: string;
  content: string;
  timestamp: number;
  read: boolean;
}

export interface MeshNotesResponse {
  notes: MeshNote[];
}

export interface WsSnapshotData {
  agents: AgentInfo[];
  memory: MeshMemoryEntry[];
  notes: MeshNote[];
}

export interface WsMessage {
  type: 'snapshot' | 'memory_written' | 'note_posted' | 'ping';
  data: WsSnapshotData | MeshMemoryEntry | MeshNote | { active_agents: number };
}

export interface ModalOptions {
  title?: string;
  message?: string;
  input?: boolean;
  inputDefault?: string;
  confirmText?: string;
  confirmClass?: string;
  onConfirm?: (value: string | boolean) => void;
}

export interface ShowConfirmOpts {
  title?: string;
  confirmText?: string;
  danger?: boolean;
}

export interface SseImportEvent {
  stage: string;
  pct: number;
  msg?: string;
  extracted?: number;
  stored?: number;
  fiction?: number;
  duplicates?: number;
}
