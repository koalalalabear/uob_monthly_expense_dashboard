import sys
from pathlib import Path

# Add root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parents[2] 
sys.path.append(str(ROOT_DIR))

import re
from datetime import datetime
import fitz        # PyMuPDF
import pandas as pd
import streamlit as st
import altair as alt


from app.inference_pipeline import run_inference


st.set_page_config(page_title="UOB Account Dashboard", layout="wide")
st.title("📄 UOB Bank Account Dashboard")


# --- 1) PDF → raw text with formatting ---
def extract_text_from_pdf(file_bytes: bytes) -> tuple[str, list]:
    """Extract text and formatting information from PDF using PyMuPDF."""
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        
        # Extract text with formatting information
        formatted_blocks = []
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            formatted_blocks.append({
                                "text": span["text"],
                                "bold": bool(span["flags"] & 2**4),  # Bold flag
                                "bbox": span["bbox"],
                                "size": span["size"]
                            })
        
        doc.close()
        return text, formatted_blocks
    except Exception as e:
        st.error(f"❌ Failed to read PDF: {e}")
        return "", []

# --- 2) Clean description text ---
def extract_alphabets(desc: str, info: str) -> str:
    # Combine and clean line breaks
    combined = f"{desc} {info}".replace("\n", " ").strip()
    # Remove patterns like "29 AUG", "02 Sep" (date format dd MMM)
    combined = re.sub(r'\b\d{2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b', '', combined, flags=re.IGNORECASE)
    # Remove "SINGAPORE SG" variants
    combined = re.sub(r"\bSINGAPORE\s+SG\b", "", combined, flags=re.IGNORECASE)
    # Remove extra spaces from removing phrases
    combined = re.sub(r'\s{2,}', ' ', combined)
    # Keep only alphanumerics and spaces
    return re.sub(r"[^A-Za-z0-9\s]+", " ", combined).strip()

# --- 3) Parse credit card transactions ---
def parse_credit_card_transactions(text: str, formatted_blocks: list, year: int) -> pd.DataFrame:
    """
    Parse UOB credit card transactions using font formatting and newline patterns
    """
    transactions = []
    
    # Method 1: Use bold text to identify transaction types
    if formatted_blocks:
        transactions = parse_with_formatting(formatted_blocks, year)
 
    # Method 2: Fallback to newline pattern analysis
    if not transactions:
        st.info("No bold formatting found, using newline pattern analysis...")
        transactions = parse_with_newline_patterns(text, year)
    
    # Method 3: Final fallback to line-by-line parsing
    if not transactions:
        st.info("Newline patterns failed, trying line-by-line parsing...")
        transactions = parse_line_by_line_v2(text, year)
    
    # Create DataFrame
    df = pd.DataFrame(transactions)
    
    if df.empty:
        return df
    
    # Ensure required columns exist
    required_cols = ["date","description", "info", "withdrawal", "deposit", "balance"]
    for col in required_cols:
        if col not in df.columns:
            if col in ["date", "description", "info", "transaction_type"]:
                df[col] = ""
            else:
                df[col] = 0.0
    
    # Parse dates
    if not df.empty and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df[df['date'].notna()].copy()  # filter out invalid dates (NaT)
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')  # convert to 'YYYY-MM-DD' string format
    
    # Clean descriptions
    if not df.empty:
        df["clean_description"] = df.apply(
            lambda r: extract_alphabets(
                str(r.get("description", "")), 
                str(r.get("info", ""))
            ), 
            axis=1
        )
    # Final data cleaning step: Remove rows where both withdrawal and deposit are 0 or null
    if not df.empty:
        df = df[~((df['deposit'].fillna(0) == 0) & (df['withdrawal'].fillna(0) == 0))]
    return df[["date", "transaction_type", "clean_description", "withdrawal", "deposit", "balance"]]

def parse_with_formatting(formatted_blocks: list, year: int) -> list:
    """Parse transactions using bold text as transaction type indicators"""
    transactions = []
    current_transaction = None
    
    for i, block in enumerate(formatted_blocks):
        text = block["text"].strip()
        if not text:
            continue
        
        # Bold text likely indicates transaction type/header
        if block["bold"] and len(text) > 3:
            # Save previous transaction
            if current_transaction:
                transactions.append(finalize_transaction(current_transaction, year))
            
            # Start new transaction
            current_transaction = {
                "transaction_type": text,
                "description_parts": [],
                "amounts": [],
                "location": ""
            }
        
        # Non-bold text is likely description or amounts
        elif current_transaction:
            # Check if this text contains amounts
            amount_matches = re.findall(r'\d{1,3}(?:,\d{3})*\.\d{2}', text)
            if amount_matches:
                current_transaction["amounts"].extend(amount_matches)
            
            # Check for location indicators
            if "SINGAPORE" in text.upper() or text.upper().endswith(" SG"):
                current_transaction["location"] = text
            else:
                # Add to description
                current_transaction["description_parts"].append(text)
    
    # Don't forget the last transaction
    if current_transaction:
        transactions.append(finalize_transaction(current_transaction, year))
    
    return transactions

def parse_with_newline_patterns(text: str, year: int) -> list:
    """Parse transactions using newline patterns and text structure"""
    transactions = []
    lines = text.split('\n')
    current_transaction = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        # Pattern 1: All caps line (likely transaction type)
        if line.isupper() and len(line) > 5 and not re.match(r'^[\d\s,.]+$', line):
            # Save previous transaction
            if current_transaction:
                transactions.append(finalize_transaction(current_transaction, year))
            
            # Start new transaction
            current_transaction = {
                "transaction_type": line,
                "description_parts": [],
                "amounts": [],
                "location": ""
            }
        
        # Pattern 2: Line starts with common transaction prefixes
        elif re.match(r'^(NETS|Misc|PAYNOW|Inward|Balance|DR-|CR-)', line, re.IGNORECASE):
            # Save previous transaction
            if current_transaction:
                transactions.append(finalize_transaction(current_transaction, year))
            
            # Start new transaction
            current_transaction = {
                "transaction_type": line,
                "description_parts": [],
                "amounts": [],
                "location": ""
            }
        
        # Pattern 3: Standalone amount line (likely end of transaction)
        elif re.match(r'^\d{1,3}(?:,\d{3})*\.\d{2}$', line) and current_transaction:
            current_transaction["amounts"].append(line)
        
        # Pattern 4: Line with location info
        elif "SINGAPORE" in line.upper() or line.upper().endswith(" SG"):
            if current_transaction:
                current_transaction["location"] = line
        
        # Pattern 5: Regular description line
        elif current_transaction:
            # Check if line contains amounts within it
            amount_matches = re.findall(r'\d{1,3}(?:,\d{3})*\.\d{2}', line)
            if amount_matches:
                current_transaction["amounts"].extend(amount_matches)
                # Remove amounts from description
                desc_line = re.sub(r'\d{1,3}(?:,\d{3})*\.\d{2}', '', line).strip()
                if desc_line:
                    current_transaction["description_parts"].append(desc_line)
            else:
                current_transaction["description_parts"].append(line)
    
    # Don't forget the last transaction
    if current_transaction:
        transactions.append(finalize_transaction(current_transaction, year))
    
    return transactions

def finalize_transaction(transaction_data: dict, year: int) -> dict:
    """Convert raw transaction data into standardized format"""
    trans_type = transaction_data.get("transaction_type", "")
    description = " ".join(transaction_data.get("description_parts", [])).strip()
    amounts = transaction_data.get("amounts", [])
    location = transaction_data.get("location", "")
    
    # Parse amounts
    parsed_amounts = [float(amt.replace(',', '')) for amt in amounts if amt]
    main_amount = parsed_amounts[0] if parsed_amounts else 0.0
    
    # Extract date from description if present
    date_match = re.search(r'(\d{1,2}\s+\w{3})', description)
    trans_date = date_match.group(1) if date_match else "01 Jan"
    
    # Determine if it's credit or debit
    is_credit = any(keyword in trans_type.lower() for keyword in [
        'inward', 'credit', 'cr-', 'deposit', 'refund', 'reversal'
    ])
    
    return {
        'date': f"{trans_date} {year}",
        'transaction_type': trans_type,
        'description': description,
        'info': location,
        'withdrawal': 0.0 if is_credit else main_amount,
        'deposit': main_amount if is_credit else 0.0,
        'balance': parsed_amounts[-1] if len(parsed_amounts) > 1 else main_amount
    }

def parse_line_by_line_v2(text: str, year: int) -> list:
    """
    Fallback parser that processes text line by line
    """
    lines = text.split('\n')
    transactions = []
    current_transaction = {}
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # Check if line contains an amount
        amount_match = re.search(r'(\d{1,3}(?:,\d{3})*\.\d{2})', line)
        
        # Check if line starts a new transaction type
        transaction_types = ['NETS', 'Misc DR', 'PAYNOW', 'Inward CR', 'Balance']
        is_transaction_start = any(line.startswith(t) for t in transaction_types)
        
        if is_transaction_start:
            # Save previous transaction if exists
            if current_transaction:
                transactions.append(current_transaction.copy())
                current_transaction = {}
            
            current_transaction['transaction_type'] = line
            current_transaction['description'] = ''
            
        elif amount_match and current_transaction:
            # This line contains the amount
            amount = float(amount_match.group(1).replace(',', ''))
            
            # Determine if it's a credit or debit
            if 'inward cr' in current_transaction.get('transaction_type', '').lower():
                current_transaction['deposit'] = amount
                current_transaction['withdrawal'] = 0.0
            else:
                current_transaction['withdrawal'] = amount
                current_transaction['deposit'] = 0.0
                
            current_transaction['balance'] = amount  # Temporary
            
        elif current_transaction:
            # Add to description
            if current_transaction['description']:
                current_transaction['description'] += ' ' + line
            else:
                current_transaction['description'] = line
    
    # Don't forget the last transaction
    if current_transaction:
        transactions.append(current_transaction)
    
    # Create list of dictionaries for consistency
    transaction_list = []
    for trans in transactions:
        # Add missing columns
        if "date" not in trans:
            trans["date"] = "01 Jan " + str(year)
        if "info" not in trans:
            trans["info"] = ""
            
        transaction_list.append(trans)
    
    return transaction_list

# --- 4) Improved Balance B/F function ---
def balance_bf(text: str, df: pd.DataFrame = None) -> float:
    """
    Extract Balance Brought Forward amount from PDF text or DataFrame.
    
    Args:
        text: Raw PDF text
        df: Optional DataFrame to search in as fallback
    
    Returns:
        float: Balance B/F amount, or 0.0 if not found
    """
    balance_bf_amount = 0.0
    
    # Method 1: Search in raw PDF text first (most reliable)
    if text:
        # Look for various patterns of "BALANCE B/F" or "BALANCE BROUGHT FORWARD"
        patterns = [
            r'BALANCE\s+B[/\\]F[^\d]*(\d{1,3}(?:,\d{3})*\.\d{2})',  # BALANCE B/F 1,234.56
            r'BALANCE\s+BROUGHT\s+FORWARD[^\d]*(\d{1,3}(?:,\d{3})*\.\d{2})',  # BALANCE BROUGHT FORWARD 1,234.56
            r'BAL\s+B[/\\]F[^\d]*(\d{1,3}(?:,\d{3})*\.\d{2})',  # BAL B/F 1,234.56
            r'B[/\\]F\s+BALANCE[^\d]*(\d{1,3}(?:,\d{3})*\.\d{2})',  # B/F BALANCE 1,234.56
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    balance_bf_amount = float(matches[0].replace(',', ''))
                    break
                except ValueError:
                    continue
    
    # Method 2: Search in DataFrame as fallback
    if balance_bf_amount == 0.0 and df is not None and not df.empty:
        # Search in clean_description column
        if 'clean_description' in df.columns:
            bf_rows = df[df["clean_description"].str.upper().str.contains("BALANCE B F|BAL B F|BALANCE BF", na=False)]
            if not bf_rows.empty and 'balance' in df.columns:
                try:
                    balance_bf_amount = float(bf_rows.iloc[0]["balance"])
                    st.info(f"Found Balance B/F in DataFrame: ${balance_bf_amount:,.2f}")
                except (ValueError, IndexError):
                    pass
        
        # Search in transaction_type column as additional fallback
        if balance_bf_amount == 0.0 and 'transaction_type' in df.columns:
            bf_rows = df[df["transaction_type"].str.upper().str.contains("BALANCE B", na=False)]
            if not bf_rows.empty:
                try:
                    balance_bf_amount = float(bf_rows.iloc[0]["balance"])
                    st.info(f"Found Balance B/F in transaction type: ${balance_bf_amount:,.2f}")
                except (ValueError, IndexError):
                    pass
    
    return balance_bf_amount

# --- 5) Extract values from End of Transaction Details section ---
def extract_transaction_totals(text: str) -> tuple[float, float, float]:
    """
    Extract the three values above 'End of Transaction Details' line.
    
    Args:
        text: Raw PDF text
    
    Returns:
        tuple: (withdrawals, deposits, balance_cf)
    """
    # Pattern to find the section before "End of Transaction Details"
    pattern = r'Total\s*\n\s*([\d,]+\.[\d]{2})\s*([\d,]+\.[\d]{2})\s*([\d,]+\.[\d]{2})\s*[-\s]*End of Transaction Details'
    
    matches = re.search(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
    
    if matches:
        try:
            withdrawals = float(matches.group(1).replace(',', ''))
            deposits = float(matches.group(2).replace(',', ''))
            balance_cf = float(matches.group(3).replace(',', ''))
            return withdrawals, deposits, balance_cf
        except (ValueError, AttributeError):
            pass
    
    # Alternative pattern - more flexible spacing
    alt_pattern = r'Total\s*\n((?:\s*[\d,]+\.[\d]{2}\s*){3}).*?End of Transaction Details'
    alt_matches = re.search(alt_pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
    
    if alt_matches:
        # Extract all numbers from the matched section
        numbers = re.findall(r'([\d,]+\.[\d]{2})', alt_matches.group(1))
        if len(numbers) >= 3:
            try:
                withdrawals = float(numbers[0].replace(',', ''))
                deposits = float(numbers[1].replace(',', ''))
                balance_cf = float(numbers[2].replace(',', ''))
                
                st.info(f"✅ Found transaction totals (alternative): Withdrawals=${withdrawals:,.2f}, Deposits=${deposits:,.2f}, Balance C/F=${balance_cf:,.2f}")
                return withdrawals, deposits, balance_cf
            except ValueError:
                pass
    
    st.warning("❌ Could not find transaction totals before 'End of Transaction Details'")
    return 0.0, 0.0, 0.0

# --- 6) Total Deposits function ---
def get_total_deposits(text: str) -> float:
    """
    Extract Total Deposits (second value) from End of Transaction Details section.
    """
    _, deposits, _ = extract_transaction_totals(text)
    return deposits

# --- 7) Total Withdrawals function ---
def get_total_withdrawals(text: str) -> float:
    """
    Extract Total Withdrawals (first value) from End of Transaction Details section.
    """
    withdrawals, _, _ = extract_transaction_totals(text)
    return withdrawals

# --- 8) Balance C/F function ---
def get_balance_cf_from_text(text: str) -> float:
    """
    Extract Balance C/F (third value) from End of Transaction Details section.
    """
    _, _, balance_cf = extract_transaction_totals(text)
    return balance_cf

# --- 9) Streamlit UI ---


# -------------- Streamlit UI --------------
uploaded_file = st.sidebar.file_uploader(
    "Upload your UOB statement PDF", type=["pdf"]
)

if uploaded_file:
    year = st.sidebar.number_input(
        "Select year", min_value=2000, max_value=2100,
        value=datetime.now().year
    )
    st.sidebar.success(f"Loaded: {uploaded_file.name}")
    st.sidebar.markdown(f"Year: `{year}`")
    
    pdf_bytes = uploaded_file.read()
    text, formatted_blocks = extract_text_from_pdf(pdf_bytes)
    
    if not text.strip():
        st.error("No text extracted – is it a scanned PDF?")
    else:
        df = parse_credit_card_transactions(text, formatted_blocks, year)
        
        if df.empty:
            st.warning("No transactions found. Check PDF format.")
        else:
            
            # Get Balance Metrics
            bf = balance_bf(text, df)
            tot_w = get_total_withdrawals(text)
            tot_d = get_total_deposits(text)
            cf = get_balance_cf_from_text(text)

            # Show Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("💰 Balance B/F", f"${bf:,.2f}")
            with col2:
                st.metric("💸 Total Withdrawals", f"${tot_w:,.2f}")
            with col3:
                st.metric("💵 Total Deposits", f"${tot_d:,.2f}")
            st.metric("📈 Balance C/F", f"${cf:,.2f}")

            # Clean up transaction DataFrame
            df_clean = df[~df["clean_description"]
                           .str.upper()
                           .str.contains("BALANCE B F")]
            mask = df_clean['transaction_type'] == 'One Bonus Interest'
            df_clean.loc[mask, 'deposit'] = df_clean.loc[mask, 'withdrawal']
            df_clean.loc[mask, 'withdrawal'] = 0
            df_clean = df_clean.reset_index(drop=True)
            df_clean = df_clean.iloc[2:]

            # --- Persist classification state ---
            if "classified" not in st.session_state:
                st.session_state.classified = False

            # --- Classification button ---
            if st.button("📂 Classify Transactions"):
                st.session_state.classified = True

            # --- If classified, run inference and display editable table ---
            if st.session_state.classified:
                st.info("Find categories & confidence below: ")

                result_df = run_inference(df_clean)

                # Merge prediction results
                df_clean = pd.concat(
                    [df_clean.reset_index(drop=True), result_df[["Predicted Category", "Confidence Score"]]],
                    axis=1,
                )

                # Drop "balance" column if it exists
                df_display = df_clean.copy()
                if 'balance' in df_display.columns:
                    df_display = df_display.drop(columns='balance')

                # Drop rows with missing 'date' or 'clean_description'
                invalid = ['date', 'clean_description']
                cols_to_check = [col for col in invalid if col in df_display.columns]
                df_display = df_display.dropna(subset=cols_to_check)

                # Clean 'clean_description' column
                df_display['clean_description'] = (
                    df_display['clean_description']
                    .str.replace(r'\d+', '', regex=True)
                    .str.strip()
                )
                df_display['clean_description'] = (
                    df_display['transaction_type'].fillna('').str.strip() + ' ' +
                    df_display['clean_description']
                )
                df_display = df_display.drop(columns='transaction_type')

                # Editable table
                df_display = st.data_editor(df_display, num_rows="dynamic", use_container_width=True)

                # --- Visualize corrected data ---
                if "show_monthly_expenses" not in st.session_state:
                    st.session_state["show_monthly_expenses"] = False

                if st.button("👁️👄👁️ Visualize Monthly Expenses"):
                    st.session_state["show_monthly_expenses"] = True

                if st.session_state["show_monthly_expenses"]:
                

                    # Prepare data and charts (your existing code)
                    df_withdrawals = df_display[df_display['withdrawal'] > 0]
                    category_stats = df_withdrawals.groupby('Predicted Category').agg(
                        transaction_count=('withdrawal', 'count'),
                        total_withdrawal=('withdrawal', 'sum')
                    ).reset_index()
                    category_stats = category_stats.sort_values('total_withdrawal', ascending=True)

                    count_chart = alt.Chart(category_stats).mark_bar(color="#77D8C5").encode(
                        x=alt.X('transaction_count:Q', title='Total Frequency', scale=alt.Scale(reverse=True)),
                        y=alt.Y('Predicted Category:N', sort=category_stats['Predicted Category'].tolist(), axis=None),
                        tooltip=['Predicted Category', 'transaction_count']
                    ).properties(width=450, height=300)

                    value_chart = alt.Chart(category_stats).mark_bar(color="#80B9DF").encode(
                        x=alt.X('total_withdrawal:Q', title='Total Value ($)'),
                        y=alt.Y('Predicted Category:N', sort=category_stats['Predicted Category'].tolist(), title=''),
                        tooltip=['Predicted Category', 'total_withdrawal']
                    ).properties(width=450, height=300)

                    # Layout side-by-side: use columns here
                    col_chart, col_metric = st.columns([4, 1])

                    with col_chart:
                        combined_chart = alt.hconcat(count_chart, value_chart).resolve_scale(y='shared')
                        st.altair_chart(combined_chart, use_container_width=True)

                    # Calculate spending ratio metric
                    df_viz = df_display.copy()
                    df_viz['withdrawal'] = pd.to_numeric(df_viz['withdrawal'], errors='coerce').fillna(0)
                    df_viz['deposit'] = pd.to_numeric(df_viz['deposit'], errors='coerce').fillna(0)
                    total_withdrawals = df_viz['withdrawal'].sum()
                    total_deposits = df_viz['deposit'].sum()

                    if total_deposits > 0:
                        pct_spent = (total_withdrawals / total_deposits) * 100
                    else:
                        pct_spent = 0

                    with col_metric:
                        st.markdown("### 📉 Spending Ratio")
                        st.markdown(f"**{pct_spent:.1f}%** of deposits spent on withdrawals")
                        if pct_spent > 100:
                            st.markdown("⚠️ You have spent more than your deposits!")

                        # Calculate no-spend days
                        df_viz['date'] = pd.to_datetime(df_viz['date'])
                        floor = df_viz.iloc[:-1]['date'].min()
                        ceilling = df_viz['date'].max()
                        all_days = pd.date_range(floor, ceilling)
                        withdrawal_days = pd.to_datetime(df_viz[df_viz['withdrawal'] > 0]['date'].unique())
                        no_spend_days = len(set(d.date() for d in all_days) - set(d.date() for d in withdrawal_days))

                        
                        # Display no-spend days
                        st.markdown(f"### 📆 No-Spend Days from **{floor.date()}** to **{ceilling.date()}**")
                        st.markdown(f"**{no_spend_days}** days with no withdrawals")


                    # Define emoji map
                    EMOJI_MAP = {
                        "Food": "🤰",
                        "Subscription":"🔄",
                        "Health":"🔆",
                        "Travel": "🛸",
                        "Groceries": "🥝",
                        "Shopping": "🛍️",
                        "Beauty": " 🎀",
                        "Fun": "😎",
                        "Others": "⁉️"
                    }

                    # Filter withdrawals
                    df_withdrawals = df_display[df_display['withdrawal'] > 0].copy()
                    df_withdrawals['category_emoji'] = df_withdrawals['Predicted Category'].map(EMOJI_MAP).fillna(df_withdrawals['Predicted Category'])

                    # Create stats first
                    category_stats = df_withdrawals.groupby('Predicted Category').agg(
                        transaction_count=('withdrawal', 'count'),
                        total_withdrawal=('withdrawal', 'sum')
                    ).reset_index()

                    # THEN map emojis to category_stats
                    category_stats['category_emoji'] = category_stats['Predicted Category'].map(EMOJI_MAP).fillna(category_stats['Predicted Category'])

                    st.subheader("Here's a look into your spending habits throughout the month, hover over the emojis for details.")
                    scatter = alt.Chart(df_withdrawals).mark_text(
                        fontSize=20,  # fixed font size for all emojis
                        fontWeight='bold',
                        baseline='middle',
                        dx=0,
                        dy=0,
                    ).encode(
                        x=alt.X('date:T', title='Date'),
                        y=alt.Y('withdrawal:Q', title='Withdrawal Amount ($)'),
                        text='category_emoji:N',  # emoji + label string

                        # Remove the size encoding below
                        # size=alt.Size('withdrawal:Q', scale=alt.Scale(range=[100, 1000]), legend=None),

                    tooltip=[
                        alt.Tooltip('date:T', title='Date'),
                        alt.Tooltip('clean_description:N', title='Description'),
                        alt.Tooltip('withdrawal:Q', title='Amount ($)'),
                        alt.Tooltip('Predicted Category:N', title='Category')]

                    ).properties(
                        width=600,
                        height=400,
                        title='Withdrawal Transactions Over Time'
                    ).interactive()

                    st.altair_chart(scatter, use_container_width=True)

                    st.subheader("Filter Transactions by Category")
                    df_withdrawals['withdrawal'] = pd.to_numeric(df_withdrawals['withdrawal'], errors='coerce')

                    category_list = sorted(df_withdrawals['Predicted Category'].unique())
                    selected_category = st.selectbox("Select a category to view transactions:", ["All"] + category_list)

                    if selected_category != "All":
                        filtered_df = df_withdrawals[df_withdrawals['Predicted Category'] == selected_category].sort_values(by='withdrawal', ascending=False)
                        merchant_counts = filtered_df['clean_description'].value_counts()
                        filtered_df['Repurchase Count'] = filtered_df['clean_description'].map(merchant_counts)
                    else:
                        filtered_df = df_withdrawals.sort_values(by='withdrawal', ascending=False)
                        merchant_counts = filtered_df['clean_description'].value_counts()
                        filtered_df['Repurchase Count'] = filtered_df['clean_description'].map(merchant_counts)

                    cols = filtered_df.columns.tolist()
                    # Remove 'Repurchase Count' if already in cols to avoid duplicates
                    if 'Repurchase Count' in cols:
                        cols.remove('Repurchase Count')

                    # Insert 'Repurchase Count' at index 3 (4th position)
                    cols.insert(3, 'Repurchase Count')

                    # Reorder dataframe columns
                    filtered_df = filtered_df[cols]
                    # Hide last 4 columns only for display
                    display_df = filtered_df.iloc[:, :-4].reset_index(drop=True)

                    st.subheader(f"Transactions in category: {selected_category}")
                    st.dataframe(display_df)

try:
    csv = df_display.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "uob_transactions.csv", "text/csv")
except NameError:
    st.info("📤 Upload a UOB PDF bank statement using the sidebar to get started.")
