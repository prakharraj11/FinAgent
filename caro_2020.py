"""
CARO 2020 – Companies (Auditor's Report) Order, 2020
=====================================================
All 21 clauses as structured data.

Each clause entry contains:
  - clause_number   : int
  - title           : str
  - legal_text      : str   (verbatim / paraphrased from MCA notification)
  - required_data   : list  (what financial data / documents are needed)
  - audit_questions : list  (specific questions the auditor must answer)
  - data_fields     : list  (exact field names to look for in financial docs)

Used by the CARO-aware auditor node to iterate clause by clause.
"""

from typing import TypedDict, List

class CAROClause(TypedDict):
    clause_number:   int
    title:           str
    legal_text:      str
    required_data:   List[str]
    audit_questions: List[str]
    data_fields:     List[str]


CARO_2020_CLAUSES: List[CAROClause] = [

    # ── Clause 1 ──────────────────────────────────────────────────────────────
    {
        "clause_number": 1,
        "title": "Property, Plant and Equipment (PPE) & Intangible Assets",
        "legal_text": (
            "Whether the company is maintaining proper records showing full particulars, "
            "including quantitative details and situation of Property, Plant and Equipment "
            "and relevant details of Right-of-Use assets; whether these assets are physically "
            "verified by management at reasonable intervals; whether any discrepancies were "
            "noticed and if so, whether they have been properly dealt with in the books; "
            "whether the title deeds of all the immovable properties are held in the name of "
            "the company — if not, provide details."
        ),
        "required_data": [
            "Fixed asset register / PPE schedule",
            "Balance sheet PPE note",
            "Right-of-use asset note",
            "Physical verification reports",
            "Title deeds / property documents",
            "Note 27(M) immovable property disclosure",
        ],
        "audit_questions": [
            "Are proper records maintained for all PPE including quantitative details?",
            "Has management physically verified PPE at reasonable intervals?",
            "Were any discrepancies noted during physical verification?",
            "Are title deeds of immovable properties held in the company's name?",
            "Are there any properties not held in the company's name — if so, are reasons given?",
            "Is the gross block / net block movement schedule complete and reconciled?",
        ],
        "data_fields": [
            "property_plant_equipment_gross",
            "property_plant_equipment_net",
            "accumulated_depreciation",
            "capital_work_in_progress",
            "right_of_use_assets",
            "intangible_assets",
            "depreciation_charge",
            "disposals",
            "additions_to_ppe",
        ],
    },

    # ── Clause 2 ──────────────────────────────────────────────────────────────
    {
        "clause_number": 2,
        "title": "Inventories & Working Capital Loans",
        "legal_text": (
            "Whether physical verification of inventory has been conducted at reasonable "
            "intervals by management and whether discrepancies of 10% or more in the aggregate "
            "for each class of inventory were noticed and properly dealt with; whether during "
            "any point of time of the year, the company has been sanctioned working capital "
            "limits in excess of five crore rupees, in aggregate, from banks or financial "
            "institutions on the basis of security of current assets — and if so, whether the "
            "quarterly returns or statements filed by the company with such banks or financial "
            "institutions are in agreement with the books of account."
        ),
        "required_data": [
            "Inventory note from balance sheet",
            "Physical verification reports for inventory",
            "Working capital loan sanction letters",
            "Quarterly stock statements filed with banks",
            "Stock statement reconciliation",
        ],
        "audit_questions": [
            "Has physical verification of inventories been done at reasonable intervals?",
            "Were discrepancies of 10% or more noticed in any class of inventory?",
            "Does the company have working capital limits > ₹5 crore from banks?",
            "Are quarterly returns filed with banks in agreement with books of account?",
            "What is the nature and value of inventories (raw material, WIP, finished goods)?",
        ],
        "data_fields": [
            "inventories_total",
            "inventories_raw_material",
            "inventories_wip",
            "inventories_finished_goods",
            "inventories_stores_spares",
            "working_capital_loan_limit",
            "current_assets_total",
        ],
    },

    # ── Clause 3 ──────────────────────────────────────────────────────────────
    {
        "clause_number": 3,
        "title": "Investments, Guarantees, Advances and Loans to Parties",
        "legal_text": (
            "Whether during the year the company has made investments in, provided any "
            "guarantee or security or granted any loans or advances in the nature of loans, "
            "secured or unsecured, to companies, firms, LLPs or other parties — and if so, "
            "whether the terms and conditions are not prejudicial to the company's interest; "
            "whether the schedule of repayment of principal and payment of interest has been "
            "stipulated and the repayments/receipts are regular; whether any amount is "
            "overdue and steps taken for recovery."
        ),
        "required_data": [
            "Loans given schedule",
            "Investments note",
            "Guarantees and securities given",
            "Related party transaction disclosure",
            "Loan agreements / terms",
            "Recovery actions for overdue loans",
        ],
        "audit_questions": [
            "Has the company granted loans/advances to subsidiaries, JVs, associates or others?",
            "Are terms and conditions of loans not prejudicial to the company's interest?",
            "Is a repayment schedule stipulated for all loans granted?",
            "Are repayments being received as per schedule?",
            "Are there overdue amounts — if so, what is the amount and recovery action taken?",
            "Has the company provided guarantees or securities on behalf of others?",
        ],
        "data_fields": [
            "loans_given_non_current",
            "loans_given_current",
            "investments_subsidiaries",
            "investments_joint_ventures",
            "investments_associates",
            "guarantees_given",
            "related_party_loans",
            "advances_to_parties",
        ],
    },

    # ── Clause 4 ──────────────────────────────────────────────────────────────
    {
        "clause_number": 4,
        "title": "Loans and Investments under Sections 185 and 186",
        "legal_text": (
            "In respect of loans, investments, guarantees and security, whether provisions "
            "of Section 185 and 186 of the Companies Act, 2013 have been complied with; "
            "if not, provide the details thereof."
        ),
        "required_data": [
            "Board resolutions for loans to directors",
            "Section 185 compliance documentation",
            "Section 186 limit calculations",
            "Investments register",
            "Guarantees register",
        ],
        "audit_questions": [
            "Are any loans given to directors or related parties under Section 185?",
            "Does the company comply with Section 186 limits for loans, guarantees and investments?",
            "Have all required approvals (board/shareholder) been obtained?",
            "Is the aggregate of loans, guarantees and investments within the prescribed limit?",
        ],
        "data_fields": [
            "loans_to_directors",
            "total_investments",
            "total_guarantees_given",
            "paid_up_capital",
            "free_reserves",
            "securities_premium",
        ],
    },

    # ── Clause 5 ──────────────────────────────────────────────────────────────
    {
        "clause_number": 5,
        "title": "Deposits",
        "legal_text": (
            "Whether the company has accepted any deposits or amounts which are deemed to be "
            "deposits within the meaning of Sections 73 to 76 or any other relevant provisions "
            "of the Companies Act, 2013 and the rules made thereunder; whether the company has "
            "complied with the provisions of Sections 73 to 76 and the rules made thereunder "
            "with regard to the deposits accepted from the public."
        ),
        "required_data": [
            "Deposit register",
            "Fixed deposit receipts",
            "Compliance certificates for deposits",
            "RBI circulars compliance (if applicable)",
        ],
        "audit_questions": [
            "Has the company accepted public deposits under Sections 73-76?",
            "If yes, has it complied with all deposit acceptance rules?",
            "Are there any amounts received that could be deemed as deposits?",
            "Has any order been received from CLB / NCLT / RBI regarding repayment of deposits?",
        ],
        "data_fields": [
            "deposits_accepted",
            "public_deposits_outstanding",
            "security_deposits_received",
            "advance_from_customers",
        ],
    },

    # ── Clause 6 ──────────────────────────────────────────────────────────────
    {
        "clause_number": 6,
        "title": "Maintenance of Cost Records",
        "legal_text": (
            "Whether maintenance of cost records has been specified by the Central Government "
            "under sub-section (1) of Section 148 of the Companies Act and whether such "
            "accounts and records have been so made and maintained."
        ),
        "required_data": [
            "CRA-1 cost accounting records",
            "Product cost statements",
            "Cost audit report (if applicable)",
            "MCA notification for the specific industry",
        ],
        "audit_questions": [
            "Is the company required to maintain cost records under Section 148(1)?",
            "Has the company maintained cost records as prescribed?",
            "Is a cost auditor appointed where required?",
        ],
        "data_fields": [
            "revenue_from_operations",
            "direct_expenses",
            "cost_of_services",
            "cost_of_goods_sold",
        ],
    },

    # ── Clause 7 ──────────────────────────────────────────────────────────────
    {
        "clause_number": 7,
        "title": "Statutory Dues",
        "legal_text": (
            "Whether the company is regular in depositing undisputed statutory dues including "
            "Goods and Services Tax, Provident Fund, Employees' State Insurance, Income Tax, "
            "Sales Tax, Service Tax, Customs Duty, Excise Duty, Value Added Tax, cess and any "
            "other statutory dues to the appropriate authorities; if not, the extent of the "
            "arrears as at the last day of the financial year concerned for a period of more "
            "than six months shall be indicated; whether any dues of GST, PF, ESI, Income Tax, "
            "Customs Duty, Excise Duty, Value Added Tax and cess have not been deposited on "
            "account of any dispute."
        ),
        "required_data": [
            "GST returns and payment challans",
            "Provident Fund payment records",
            "ESI payment records",
            "Income Tax payment records / Form 26AS",
            "TDS returns",
            "Contingent liabilities note for disputed dues",
            "Statutory dues outstanding schedule",
        ],
        "audit_questions": [
            "Are all statutory dues (GST, PF, ESI, TDS, Income Tax) paid on time?",
            "Are there any undisputed statutory dues outstanding for more than 6 months?",
            "What is the amount of disputed statutory dues pending before forums?",
            "Are all GST returns filed on time with correct amounts?",
            "Is there any TDS default or short deduction?",
        ],
        "data_fields": [
            "statutory_dues_gst",
            "statutory_dues_tds",
            "statutory_dues_pf",
            "statutory_dues_esi",
            "income_tax_payable",
            "income_tax_contingent",
            "contingent_liabilities_tax",
            "contingent_liabilities_total",
            "other_liabilities_statutory",
        ],
    },

    # ── Clause 8 ──────────────────────────────────────────────────────────────
    {
        "clause_number": 8,
        "title": "Unrecorded Income / Transactions Surrendered to Tax Authorities",
        "legal_text": (
            "Whether any transactions not recorded in the books of account have been "
            "surrendered or disclosed as income during the year in the tax assessments under "
            "the Income Tax Act, 1961; if so, whether the previously unrecorded income has "
            "been properly recorded in the books of account during the year."
        ),
        "required_data": [
            "Income tax assessment orders",
            "Tax audit report (Form 3CA/3CB & 3CD)",
            "Disclosure of income during search/survey",
            "Note on undisclosed income in financial statements",
        ],
        "audit_questions": [
            "Has any previously undisclosed income been surrendered to tax authorities?",
            "Has such income been properly recorded in the books of account?",
            "Is there any income surrendered during search/survey not reflected in accounts?",
        ],
        "data_fields": [
            "other_income_total",
            "current_tax_expense",
            "deferred_tax_asset",
            "deferred_tax_liability",
            "income_tax_contingent",
        ],
    },

    # ── Clause 9 ──────────────────────────────────────────────────────────────
    {
        "clause_number": 9,
        "title": "Borrowings, Default and End-Use of Funds",
        "legal_text": (
            "Whether the company has defaulted in repayment of loans or other borrowings or "
            "in the payment of interest thereon to any lender; whether the company is a "
            "declared wilful defaulter by any bank or financial institution or government or "
            "any government authority; whether term loans were applied for the purpose for "
            "which the loans were obtained; whether funds raised on short-term basis have been "
            "utilised for long-term purposes; whether funds raised for specific purposes have "
            "been utilised for those purposes only."
        ),
        "required_data": [
            "Loan agreements and term sheets",
            "Bank statements",
            "Wilful defaulter certificate / declaration",
            "End-use certificates from banks",
            "Cash flow statement (borrowings section)",
            "Note 27(N) wilful defaulter disclosure",
        ],
        "audit_questions": [
            "Has the company defaulted on any loan repayment or interest payment?",
            "Is the company a declared wilful defaulter?",
            "Have term loans been used for the purposes for which they were sanctioned?",
            "Have short-term funds been deployed for long-term use?",
            "Is there any diversion of funds raised for specific purposes?",
        ],
        "data_fields": [
            "long_term_borrowings",
            "short_term_borrowings",
            "finance_costs",
            "interest_expense",
            "loan_repayments",
            "lease_liabilities",
            "net_debt",
        ],
    },

    # ── Clause 10 ─────────────────────────────────────────────────────────────
    {
        "clause_number": 10,
        "title": "Moneys Raised by Issue of Securities",
        "legal_text": (
            "Whether moneys raised by way of initial public offer or further public offer "
            "(including debt instruments) and/or term loans were applied for the purposes for "
            "which those were raised; whether the company has made any preferential allotment "
            "or private placement of shares or convertible debentures or convertible preference "
            "shares or fully or partly convertible debentures during the year."
        ),
        "required_data": [
            "Prospectus / offer document",
            "Share allotment records",
            "End-use statement for IPO / FPO proceeds",
            "Debenture trust deed",
            "Board resolution for preferential allotment",
        ],
        "audit_questions": [
            "Were IPO / FPO / NCD proceeds utilised for stated purposes?",
            "Has any preferential allotment or private placement occurred during the year?",
            "Were all SEBI / MCA compliances met for any such issuance?",
        ],
        "data_fields": [
            "equity_share_capital",
            "securities_premium",
            "share_application_money",
            "proceeds_from_share_issue",
            "debentures_issued",
        ],
    },

    # ── Clause 11 ─────────────────────────────────────────────────────────────
    {
        "clause_number": 11,
        "title": "Fraud",
        "legal_text": (
            "Whether any fraud by the company or any fraud on the company has been noticed "
            "or reported during the year; if yes, the nature and the amount involved is "
            "to be indicated; whether any report under sub-section (12) of Section 143 of "
            "the Companies Act has been filed by the auditors in Form ADT-4 with the Central "
            "Government; whether the auditor has considered whistle-blower complaints "
            "received during the year by the company."
        ),
        "required_data": [
            "Fraud register / fraud incident reports",
            "Internal audit reports for fraud indicators",
            "Whistle-blower complaints register",
            "ADT-4 filing records",
            "Board minutes on fraud incidents",
        ],
        "audit_questions": [
            "Has any fraud been noticed or reported during the year?",
            "What is the nature and amount of any fraud?",
            "Has the auditor filed ADT-4 with Central Government for any fraud?",
            "Were whistle-blower complaints received and how were they handled?",
            "Are there any fraud risk indicators in the financial data?",
        ],
        "data_fields": [
            "exceptional_items",
            "prior_period_items",
            "provisions_for_doubtful_debts",
            "write_offs",
            "litigation_contingencies",
        ],
    },

    # ── Clause 12 ─────────────────────────────────────────────────────────────
    {
        "clause_number": 12,
        "title": "Nidhi Company",
        "legal_text": (
            "Whether the Nidhi Company has complied with the Net Owned Funds to Deposits "
            "ratio of 1:20 to meet out the liability and whether the Nidhi Company is "
            "maintaining ten percent unencumbered term deposits as specified in the Nidhi Rules."
        ),
        "required_data": [
            "Nidhi Company registration certificate",
            "Net owned fund calculation",
            "Deposit register",
        ],
        "audit_questions": [
            "Is the company a Nidhi Company?",
            "If yes, is the NOF to Deposits ratio within 1:20?",
            "Are 10% unencumbered term deposits maintained?",
        ],
        "data_fields": [
            "company_type",
            "net_owned_funds",
            "total_deposits",
        ],
    },

    # ── Clause 13 ─────────────────────────────────────────────────────────────
    {
        "clause_number": 13,
        "title": "Related Party Transactions",
        "legal_text": (
            "Whether all transactions with the related parties are in compliance with "
            "Sections 177 and 188 of the Companies Act, 2013 where applicable and the "
            "details have been disclosed in the Financial Statements as required by "
            "the applicable accounting standards."
        ),
        "required_data": [
            "Related party transaction note (Ind AS 24)",
            "Audit committee approvals for RPTs",
            "Section 188 register of contracts",
            "Transfer pricing documentation",
            "RPT policy of the company",
        ],
        "audit_questions": [
            "Are all related party transactions properly disclosed in notes?",
            "Have Section 177 and 188 compliances been met for RPTs?",
            "Are RPTs on arm's length basis?",
            "Is Audit Committee approval obtained for all material RPTs?",
            "Are there any undisclosed related party transactions?",
        ],
        "data_fields": [
            "related_party_trade_receivables",
            "related_party_trade_payables",
            "related_party_loans",
            "related_party_investments",
            "related_party_revenue",
            "related_party_purchases",
            "dividend_paid_to_holding",
            "remuneration_to_kmp",
        ],
    },

    # ── Clause 14 ─────────────────────────────────────────────────────────────
    {
        "clause_number": 14,
        "title": "Internal Audit",
        "legal_text": (
            "Whether the company has an internal audit system commensurate with the size and "
            "nature of its business and whether the reports of the internal auditors for the "
            "period under audit were considered by the statutory auditor."
        ),
        "required_data": [
            "Internal audit reports",
            "Internal auditor appointment letter",
            "Audit committee minutes reviewing internal audit",
            "Internal audit plan and scope",
        ],
        "audit_questions": [
            "Is there a functioning internal audit system?",
            "Is the internal audit scope commensurate with the size and nature of business?",
            "Were internal audit reports reviewed by statutory auditors?",
            "Were significant internal audit findings resolved?",
        ],
        "data_fields": [
            "audit_fee_internal",
            "total_revenue",
            "total_assets",
        ],
    },

    # ── Clause 15 ─────────────────────────────────────────────────────────────
    {
        "clause_number": 15,
        "title": "Non-Cash Transactions with Directors",
        "legal_text": (
            "Whether the company has entered into any non-cash transactions with its "
            "directors or persons connected with him and if so, whether the provisions "
            "of Section 192 of the Companies Act have been complied with."
        ),
        "required_data": [
            "Non-cash transaction register",
            "Board minutes on non-cash transactions",
            "Section 192 compliance records",
        ],
        "audit_questions": [
            "Has the company entered into non-cash transactions with directors?",
            "If yes, has Section 192 been complied with (valuation, disclosure, approval)?",
        ],
        "data_fields": [
            "non_cash_transactions",
            "transactions_with_directors",
        ],
    },

    # ── Clause 16 ─────────────────────────────────────────────────────────────
    {
        "clause_number": 16,
        "title": "Registration under RBI Act",
        "legal_text": (
            "Whether the company is required to be registered under Section 45-IA of the "
            "Reserve Bank of India Act, 1934 and if so, whether the registration has been "
            "obtained."
        ),
        "required_data": [
            "RBI registration certificate (if NBFC)",
            "Certificate of commencement of NBFC business",
        ],
        "audit_questions": [
            "Is the company a Non-Banking Financial Company (NBFC)?",
            "If yes, is it registered with RBI under Section 45-IA?",
            "Is the company conducting any financial activities requiring RBI registration?",
        ],
        "data_fields": [
            "company_type",
            "financial_assets_ratio",
            "financial_income_ratio",
        ],
    },

    # ── Clause 17 ─────────────────────────────────────────────────────────────
    {
        "clause_number": 17,
        "title": "Cash Losses",
        "legal_text": (
            "Whether the company has incurred cash losses in the financial year and in the "
            "immediately preceding financial year and if so, the amount of cash losses."
        ),
        "required_data": [
            "Cash flow statement",
            "Profit and Loss account",
            "Prior year cash flow statement",
        ],
        "audit_questions": [
            "Has the company incurred cash losses in the current year?",
            "Were there cash losses in the immediately preceding year?",
            "What is the quantum of cash losses in both years?",
        ],
        "data_fields": [
            "profit_after_tax_current",
            "profit_after_tax_previous",
            "depreciation_amortization",
            "operating_cash_flow",
            "net_cash_from_operations",
        ],
    },

    # ── Clause 18 ─────────────────────────────────────────────────────────────
    {
        "clause_number": 18,
        "title": "Resignation of Statutory Auditors",
        "legal_text": (
            "Whether there has been any resignation of the statutory auditors during the "
            "year and if so, whether the auditor has taken into consideration the issues, "
            "objections or concerns raised by the outgoing auditors."
        ),
        "required_data": [
            "Auditor appointment / resignation records",
            "ADT-1 filings",
            "Outgoing auditor's communication on resignation",
        ],
        "audit_questions": [
            "Has the statutory auditor resigned during the year?",
            "If yes, what were the reasons and were all concerns addressed?",
            "Were resignation-related disclosures made to members and RoC?",
        ],
        "data_fields": [
            "statutory_audit_fee",
            "auditor_name",
        ],
    },

    # ── Clause 19 ─────────────────────────────────────────────────────────────
    {
        "clause_number": 19,
        "title": "Ability to Continue as a Going Concern",
        "legal_text": (
            "On the basis of the financial ratios, ageing and expected dates of realisation "
            "of financial assets and payment of financial liabilities, other information "
            "accompanying the financial statements, the auditor's knowledge of the Board of "
            "Directors and management plans — whether the auditor is of the opinion that no "
            "material uncertainty exists as on the date of audit report that company is "
            "capable of meeting its liabilities existing at the date of balance sheet as and "
            "when they fall due within a period of one year from the balance sheet date."
        ),
        "required_data": [
            "Ageing schedule of debtors and payables",
            "Cash flow projections",
            "Board's assessment of going concern",
            "Current ratio and working capital analysis",
            "Net worth computation",
        ],
        "audit_questions": [
            "Is the current ratio above 1 indicating ability to meet short-term liabilities?",
            "Does the company have positive net worth?",
            "Is operating cash flow consistently positive?",
            "Are there any significant going concern risks?",
            "What is the company's liquidity position?",
        ],
        "data_fields": [
            "current_assets_total",
            "current_liabilities_total",
            "current_ratio",
            "net_cash_from_operations",
            "total_equity",
            "net_profit_loss",
            "working_capital",
            "debt_equity_ratio",
        ],
    },

    # ── Clause 20 ─────────────────────────────────────────────────────────────
    {
        "clause_number": 20,
        "title": "Top 10 Financial Liability Exposure",
        "legal_text": (
            "Whether, in case of a company other than a financial company, the company "
            "has transferred the unspent amount under sub-section (5) of Section 135 of the "
            "Companies Act to the Fund specified in Schedule VII of the Companies Act within "
            "a period of six months of the expiry of the financial year in compliance with "
            "the second proviso to sub-section (5) of Section 135 of the Companies Act."
        ),
        "required_data": [
            "CSR committee report",
            "CSR expenditure note (Note 27(O))",
            "Unspent CSR balance transfer records",
            "Schedule VII fund transfer challan",
        ],
        "audit_questions": [
            "What is the prescribed CSR spend for the year (2% of average net profit)?",
            "Was the actual CSR expenditure equal to or more than the prescribed amount?",
            "Has any unspent CSR amount been transferred to the Schedule VII fund?",
            "Was unspent CSR transferred within 6 months of financial year end?",
        ],
        "data_fields": [
            "csr_expenditure_required",
            "csr_expenditure_actual",
            "csr_unspent_balance",
            "average_net_profit_3yr",
            "profit_before_tax",
        ],
    },

    # ── Clause 21 ─────────────────────────────────────────────────────────────
    {
        "clause_number": 21,
        "title": "Qualifications in Group Audit Reports",
        "legal_text": (
            "Whether the auditor has considered qualifications or adverse remarks made by "
            "the respective auditors in the reports on the financial statements of the "
            "subsidiary companies, associate companies and joint ventures incorporated in India."
        ),
        "required_data": [
            "Subsidiary audit reports",
            "Associate company audit reports",
            "Joint venture audit reports",
            "Consolidated financial statement notes",
            "List of subsidiaries, associates, JVs",
        ],
        "audit_questions": [
            "Are there any qualifications in subsidiary / associate / JV audit reports?",
            "Have all such qualifications been considered in the group audit?",
            "Are there adverse remarks that need to be reported in the parent company audit?",
            "Is the basis of consolidation appropriate?",
        ],
        "data_fields": [
            "investments_subsidiaries",
            "investments_joint_ventures",
            "investments_associates",
            "subsidiary_list",
            "dividend_from_subsidiaries",
        ],
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────────────────────────────────────

def get_clause(clause_number: int) -> CAROClause | None:
    """Return a single CARO clause by number."""
    for clause in CARO_2020_CLAUSES:
        if clause["clause_number"] == clause_number:
            return clause
    return None


def get_all_required_data_fields() -> list[str]:
    """Aggregate all unique data_fields across all 21 clauses."""
    fields = set()
    for clause in CARO_2020_CLAUSES:
        fields.update(clause["data_fields"])
    return sorted(fields)


def get_clause_summary() -> str:
    """Return a compact summary string of all 21 clauses for prompt injection."""
    lines = []
    for c in CARO_2020_CLAUSES:
        lines.append(f"Clause {c['clause_number']:02d}: {c['title']}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(f"CARO 2020 — {len(CARO_2020_CLAUSES)} clauses loaded")
    print("\nAll clauses:")
    print(get_clause_summary())
    print(f"\nTotal unique data fields required: {len(get_all_required_data_fields())}")