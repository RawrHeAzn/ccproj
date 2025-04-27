// src/types/dashboardData.ts

export interface TopSpender {
  Hshd_num: number;
  total_spend: number;
}

export interface LoyaltyTrend {
  Loyalty_flag: string; // Ensure this is string '0' or '1'
  Year: number;
  Week_num: number;
  total_spend: number;
}

export interface EngagementByIncome {
  income_bracket: string;
  avg_spend: number;
}

export interface BrandPreference {
  brand_type: string;
  total_spend: number;
}

export interface FrequentPair {
  item1: string;
  item2: string;
  count: number;
}

export interface PopularProduct {
  commodity: string;
  total_spend: number;
}

export interface SeasonalTrend {
  year: number;
  month: number;
  total_spend: number;
}

// Added Churn Risk Type
export interface ChurnRiskCustomer {
  Hshd_num: number;
  LastPurchaseDate: string; // Date as string
  Loyalty_flag: string;
  IncomeRange: string; // Assuming it might be a string category from DB
  HshdSize: number;
  Children: number;
}

// Type for summary counts
export interface SummaryCount {
    [key: string]: any; // e.g., loyalty_flag: string or income_range: string
    count: number;
}

// Updated structure for the /churn-risk endpoint response
export interface ChurnRiskData {
    at_risk_list: ChurnRiskCustomer[];
    summary_stats: {
        count_by_loyalty?: SummaryCount[];
        count_by_income?: SummaryCount[];
        error?: string;
    };
}

// --- NEW: Type for Association Rules ---
export interface AssociationRule {
  antecedents: string; // Comma-separated string of items
  consequents: string; // Comma-separated string of items
  support: number;
  confidence: number;
  lift: number;
} 