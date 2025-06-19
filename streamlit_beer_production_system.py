def sync_batch_shelf_lives(batches):
    """Ensure batch 1's shelf life is always one day less than batch 2."""
    if not batches:
        return batches
        
    for i in range(len(batches)):
        if batches[i]["batch_no"] == 1:
            # Find batch 2 if it exists
            batch2 = next((b for b in batches if b["batch_no"] == 2), None)
            if batch2:
                # Set batch 1's shelf life to one less than batch 2's
                batches[i]["shelf_life_left"] = max(0, batch2["shelf_life_left"] - 1)
            break
    return batches
                
import pickle
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, timedelta
import base64

# Ensure using the Agg backend for matplotlib
matplotlib.use('Agg')

from Alpha_DB import *  # Assuming Alpha_DB has the necessary functions

# Initialize database
init_db()

# Initialize session state variables
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'department' not in st.session_state:
    st.session_state['department'] = None
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "home"

# File to store data (ensure this is a persistent file, not in-memory)
DATA_FILE = "inventory_management.pkl"


def save_data(data):
    """Save data to a pickle file."""
    with open(DATA_FILE, "wb") as f:
        pickle.dump(data, f)

def load_data():
    """Load data from a pickle file."""
    try:
        with open(DATA_FILE, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        # System-wide persistent structure for all dashboards
        return {
            "predictions": {},
            "actual_sales": {},
            "production_targets": {},
            "last_forecast": None,
            "last_forecast_df": None,
            "production_records": None,
            "production_batches": None,
            "batch_counter": None,
            "management_reports": None,
            # Add more keys here as needed for other dashboards' persistent data
        }


def determine_income_period(date):
    """Determine income period based on the day of the month."""
    # Example: 1 for first half, 2 for second half
    if date.day <= 15:
        return 1
    else:
        return 2


def load_models():
    """Load both low and high income period models."""
    try:
        model_low = pickle.load(open('model_low_converted.pkl', 'rb'))
    except FileNotFoundError:
        st.error("Model file 'model_low_converted.pkl' not found. Please ensure the model file exists.")
        model_low = None

    try:
        model_high = pickle.load(open('model_high_converted.pkl', 'rb'))
    except FileNotFoundError:
        st.error("Model file 'model_high_converted.pkl' not found. Please ensure the model file exists.")
        model_high = None

    return model_low, model_high


def make_prediction(date, model_low, model_high):
    income_period = determine_income_period(date)
    features = pd.DataFrame({
        'day_of_month': [date.day],
        'month': [date.month],
        'year': [date.year],
        'income_period': [income_period]
    })

    expected_features = ['day_of_month', 'month', 'year', 'income_period']
    features = features[expected_features]

    if income_period == 1 and model_low is not None:
        prediction = model_low.predict(features)
    elif income_period == 2 and model_high is not None:
        prediction = model_high.predict(features)
    else:
        return None

    return round(prediction[0])
        

def generate_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def verify_hashes(password, hashed_text):
    if generate_hashes(password) == hashed_text:
        return hashed_text
    return False


def signup():
    st.sidebar.header("Signup Section")
    new_username = st.sidebar.text_input("New Username")
    new_password = st.sidebar.text_input("New Password", type="password")

    # Add department selection
    department_options = ["sales", "production", "management"]
    new_department = st.sidebar.selectbox("Department", department_options)

    if st.sidebar.button("Signup"):
        create_usertable()  # Ensure the user table exists
        hashed_new_password = generate_hashes(new_password)
        add_userdata(new_username, hashed_new_password, new_department)
        st.sidebar.success("Signup successful! You can now log in.")


def login():
    st.sidebar.header("Login Section")
    username = st.sidebar.text_input("Username").strip()
    password = st.sidebar.text_input("Password", type="password").strip()

    if st.sidebar.button("Login"):
        if not username or not password:
            st.sidebar.error("Please enter both username and password.")
            return

        create_usertable()  # Ensure the user table exists
        hashed_password = generate_hashes(password)
        # Pass hashed_password directly to login_user
        result = login_user(username, hashed_password)

        if result:
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            # Get the user's department from the database
            st.session_state['department'] = get_user_department(username)
            st.success(f"Welcome {username} ({st.session_state['department']})")
            # Restore approval_status_history and quantity_to_be_produced_map after login
            data = load_data()
            if "approval_status_history" in data and data["approval_status_history"] is not None:
                st.session_state["approval_status_history"] = data["approval_status_history"]
            if "quantity_to_be_produced_map" in data and data["quantity_to_be_produced_map"] is not None:
                st.session_state["quantity_to_be_produced_map"] = data["quantity_to_be_produced_map"]
        else:
            st.sidebar.error("Invalid username or password.")


def create_pdf_report(data, title):
    """Create a PDF report with the given data and title."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Add title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, title)
    c.setFont("Helvetica", 12)

    # Add date
    report_date = datetime.now().strftime("%Y-%m-%d")
    c.drawString(50, height - 80, f"Report Date: {report_date}")

    # Add data
    y_position = height - 120
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_position, "Report Data:")
    y_position -= 20

    c.setFont("Helvetica", 10)
    for key, value in data.items():
        c.drawString(50, y_position, f"{key}: {value}")
        y_position -= 20

        if y_position < 50:  # Check if we need a new page
            c.showPage()
            c.setFont("Helvetica", 10)
            y_position = height - 50

    # Final processing
    c.save()
    buffer.seek(0)
    return buffer


def download_pdf_button(data, title, button_text="Download PDF Report"):
    """Create a download button for a PDF report."""
    pdf = create_pdf_report(data, title)
    b64_pdf = base64.b64encode(pdf.read()).decode()

    file_name = f"{title.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf"

    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{file_name}">Download PDF Report</a>'

    st.markdown(href, unsafe_allow_html=True)


def sales_department(model_low, model_high):
    # Add background image and welcome message
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('traditional20%25flavour.jpg');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("WELCOME TO THE SALES DEPARTMENT DASHBOARD")

    # Add a reset button to clear all data (predictions, actual sales, production targets)
    if st.button("Reset All Sales Data", type="primary"):
        save_data({
            "predictions": {},
            "actual_sales": {},
            "production_targets": {},
            "last_forecast": None,
            "last_forecast_df": None
        })
        st.session_state.pop("production_records", None)
        st.session_state.pop("production_batches", None)
        st.session_state.pop("batch_counter", None)
        st.success("All sales and production data has been reset. Please refresh or rerun the app.")
        st.stop()

    # Create tabs for different functionalities
    # Remove "Stock Status" tab
    tabs = st.tabs(["Demand Forecasting", "Sales Upload", "Sales Reports"])

    with tabs[0]:  # Demand Forecasting
        st.header("Demand Forecasting")

        data = load_data()
        # Show last forecast info if available
        if data.get("last_forecast") and data.get("last_forecast_df") is not None:
            st.info(
                f"Last Forecast: {data['last_forecast']['start_date']} to {data['last_forecast']['end_date']}"
            )
            last_df = pd.DataFrame(data["last_forecast_df"])
            st.dataframe(last_df)
            plt.plot(pd.to_datetime(last_df['Date']), last_df['Predicted Sales'], marker='o')
            plt.title("Last Sales Predictions")
            plt.xlabel("Date")
            plt.ylabel("Predicted Units")
            plt.grid(True)
            plt.xticks(rotation=45)
            fig = plt.gcf()
            st.pyplot(fig)

        # Check if all actual sales have been filled for the last forecast period
        last_forecast = data.get("last_forecast")
        last_forecast_df = data.get("last_forecast_df")
        can_run_new_forecast = True
        if last_forecast and last_forecast_df:
            forecast_dates = [row["Date"] if isinstance(row["Date"], str) else row["Date"].strftime("%Y-%m-%d") for row in last_forecast_df]
            actual_sales = data.get("actual_sales", {})
            # Check if all forecast dates have actual sales filled
            can_run_new_forecast = all(
                str(actual_sales.get(date, "")).strip() != "" and str(actual_sales.get(date, "")).isdigit()
                for date in forecast_dates
            )

        if not can_run_new_forecast:
            st.warning("Please fill in all actual sales for the current forecast period before running a new forecast.")

        else:
            # Date range input for predictions (only show if all previous actuals are filled)
            pred_start_date = st.date_input("Select Start Date for Predictions")
            pred_end_date = st.date_input("Select End Date for Predictions",
                                          value=pred_start_date + timedelta(days=7))

            # Validate date inputs
            if pred_start_date > pred_end_date:
                st.error("Start date must be before end date.")
            else:
                run_forecast = st.button("Run Forecast")
                if run_forecast:
                    data = load_data()
                    if model_low is None or model_high is None:
                        st.error("Model files not found. Please ensure the model files exist.")
                    else:
                        data["predictions"] = {}

                        with st.spinner("Generating predictions..."):
                            previous_prediction = None
                            margin = 6  # minimum margin between predictions
                            for date in pd.date_range(pred_start_date, pred_end_date):
                                predicted_sale = make_prediction(date, model_low, model_high)
                                if previous_prediction is not None:
                                    if abs(predicted_sale - previous_prediction) < margin:
                                        if predicted_sale > previous_prediction:
                                            predicted_sale = previous_prediction + margin
                                        else:
                                            predicted_sale = previous_prediction - margin
                                data["predictions"][date.strftime("%Y-%m-%d")] = predicted_sale
                                previous_prediction = predicted_sale

                        # Fix for Pylance error: ensure pred_end_date is not None
                    if pred_end_date is not None:
                        pred_df = pd.DataFrame(list(data["predictions"].items()),
                                               columns=["Date", "Predicted Sales"])
                        pred_df['Date'] = pd.to_datetime(pred_df['Date'])
                        pred_df = pred_df.sort_values('Date')

                        # Save last forecast info and table for persistence
                        data["last_forecast"] = {
                            "start_date": pred_start_date.strftime("%Y-%m-%d"),
                            "end_date": pred_end_date.strftime("%Y-%m-%d"),
                        }
                        data["last_forecast_df"] = pred_df.to_dict(orient="records")
                        save_data(data)

                        st.success("Predictions generated successfully!")
                        st.dataframe(pred_df)

                        plt.plot(pred_df['Date'], pred_df['Predicted Sales'], marker='o')
                        plt.title("Sales Predictions")
                        plt.xlabel("Date")
                        plt.ylabel("Predicted Units")
                        plt.grid(True)
                        plt.xticks(rotation=45)
                        fig = plt.gcf()
                        st.pyplot(fig)

                        # Allow download of predictions as CSV
                        csv = pred_df.to_csv(index=False)
                        st.download_button(
                            "Download Predictions as CSV",
                            data=csv,
                            file_name="sales_predictions.csv",
                            mime="text/csv"
                        )

                        # Generate report data
                        report_data = {
                            "Start Date": pred_start_date.strftime("%Y-%m-%d"),
                            "End Date": pred_end_date.strftime("%Y-%m-%d"),
                            "Average Daily Prediction": pred_df["Predicted Sales"].mean(),
                            "Total Predicted Units": pred_df["Predicted Sales"].sum(),
                            "Maximum Daily Prediction": pred_df["Predicted Sales"].max(),
                            "Minimum Daily Prediction": pred_df["Predicted Sales"].min()
                        }

                        # Create PDF report download
                        download_pdf_button(report_data, "Sales Prediction Report")

                        expiring_stock = get_expiring_stock(days_threshold=7)
                        if expiring_stock:
                            expiring_df = pd.DataFrame(expiring_stock, columns=["Production Date", "Quantity", "Expiry Date"])
                            expiring_df['Production Date'] = pd.to_datetime(expiring_df['Production Date'])
                            expiring_df['Expiry Date'] = pd.to_datetime(expiring_df['Expiry Date'])
                            expiring_df = expiring_df.sort_values('Expiry Date')

                            st.dataframe(expiring_df)

                            # Total expiring quantity
                            total_expiring = expiring_df["Quantity"].sum()
                            st.error(f"Total expiring: {total_expiring} units")
                        else:
                            st.success("No stock is expiring within the next 7 days.")
            
    # --- Sales Upload Section ---
    with tabs[1]:  # Sales Upload
        st.header("Daily Sales Upload")
        data = load_data()
        if data.get("predictions"):
            pred_items = sorted(data["predictions"].items())
            actual_sales = data.get("actual_sales", {})

            # State for edit mode
            if 'edit_sales_mode' not in st.session_state:
                st.session_state['edit_sales_mode'] = False

            # --- Edit Mode ---
            if st.session_state['edit_sales_mode']:
                date_options = [date_str for date_str, _ in pred_items]
                edit_date = st.selectbox("Select Date to Edit Sales", date_options, key="edit_date_select")
                predicted_sales = data["predictions"][edit_date]
                prev_sales = actual_sales.get(edit_date, "")

                st.write(f"Predicted Sales: {predicted_sales}")
                edit_sales_val = st.number_input(
                    "Edit Actual Sales", min_value=0,
                    value=int(prev_sales) if str(prev_sales).isdigit() else 0,
                    step=1, key="edit_sales_input"
                )

                if st.button("Save Edited Sales Data", key="save_edited_sales_button"):
                    data["actual_sales"][edit_date] = int(edit_sales_val)
                    save_data(data)
                    st.success(f"Sales data for {edit_date} updated successfully!")
                    st.session_state['edit_sales_mode'] = False
                    st.rerun()
                st.stop()

            # --- Normal Sequential Entry Mode ---
            pred_dates = [pd.to_datetime(date_str) for date_str, _ in pred_items]
            date_options = [d.strftime("%Y-%m-%d") for d in pred_dates]

            # Find the first date not yet filled
            next_idx = 0
            for i, date_str in enumerate(date_options):
                if not (date_str in actual_sales and str(actual_sales[date_str]).strip() != ""):
                    next_idx = i
                    break
            else:
                next_idx = len(date_options) - 1 # All filled, stay at last

            selected_date_str = date_options[next_idx]
            st.markdown(f"**Filling sales for date:** `{selected_date_str}`")
            predicted_sales = data["predictions"][selected_date_str]
            prev_sales = actual_sales.get(selected_date_str, "")

            # Check for missing prior dates
            missing_prior = []
            for i in range(next_idx):
                prior_date = date_options[i]
                if not (prior_date in actual_sales and str(actual_sales[prior_date]).strip() != ""):
                    missing_prior.append(prior_date)

            st.write(f"Predicted Sales: {predicted_sales}")
            sales_val = st.number_input(
                "Enter Actual Sales", min_value=0,
                value=int(prev_sales) if str(prev_sales).isdigit() else 0,
                step=1, key="sales_input"
            )

            col_save, col_edit = st.columns([2, 1])
            with col_save:
                save_clicked = st.button("Save Sales Data", key="save_sales_button")
            with col_edit:
                edit_clicked = st.button("Edit", key="edit_sales_button_inline")

            if edit_clicked:
                st.session_state['edit_sales_mode'] = True
                st.rerun()

            if save_clicked:
                if missing_prior:
                    st.warning(f"You must fill sales for all prior dates before entering sales for {selected_date_str}. Missing: {', '.join(missing_prior)}")
                else:
                    data["actual_sales"][selected_date_str] = int(sales_val)
                    save_data(data)
                    st.success(f"Sales data for {selected_date_str} saved successfully!")
                    if next_idx < len(date_options) - 1:
                        st.rerun()

            summary = []
            for date_str, pred in pred_items:
                actual = data["actual_sales"].get(date_str, "")
                summary.append({"Date": date_str, "Predicted Sales": pred, "Sales": actual})
            st.dataframe(pd.DataFrame(summary))
        else:
            st.info("No predictions found. Please run a forecast first.")

    # --- Sales Reports Section ---
    with tabs[2]:  # Sales Reports
        st.header("Sales Reports")

        data = load_data()
        pred_items = sorted(data.get("predictions", {}).items())
        actual_sales = data.get("actual_sales", {})

        sales_table = []
        for date_str, pred in pred_items:
            actual = actual_sales.get(date_str, "")
            actual_val = int(actual) if str(actual).isdigit() else None
            sales_table.append({"Date": date_str, "Predicted Sales": pred, "Sales": actual_val})
        sales_df = pd.DataFrame(sales_table)

        st.subheader("Sales Table")
        st.dataframe(sales_df)

        total_predicted = sum([pred for _, pred in pred_items])
        total_actual = sum([int(actual_sales[date_str]) for date_str, _ in pred_items if str(actual_sales.get(date_str, "")).strip() != ""])
        remaining_predicted = total_predicted - total_actual
        units_to_be_sold = remaining_predicted

        st.markdown(f"**Total Predicted Sales:** {total_predicted}")
        st.markdown(f"**Total Actual Sales (to date):** {total_actual}")
        if units_to_be_sold < 0:
            st.markdown(f"**Units to be Sold:** SALES DEMAND UNDERESTIMATED")
        else:
            st.markdown(f"**Units to be Sold:** {units_to_be_sold}")

        st.subheader("Download Comprehensive Report")
        csv = sales_df.to_csv(index=False)
        st.download_button(
            "Download Sales Report as CSV",
            data=csv,
            file_name="comprehensive_sales_report.csv",
            mime="text/csv"
        )

        report_data = {
            "Total Predicted Sales": total_predicted,
            "Total Actual Sales (to date)": total_actual,
            "Remaining Predicted Sales": remaining_predicted,
            "Units to be Sold (difference)": units_to_be_sold
        }
        download_pdf_button(report_data, "Comprehensive Sales Report")

def production_department():
    # Add background image and welcome message
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('traditional20%25flavour.jpg');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("WELCOME TO THE PRODUCTION DEPARTMENT DASHBOARD")

    data = load_data()
    
    # Restore persistent production records and batches if present
    if data.get("production_records") is not None:
        st.session_state["production_records"] = data["production_records"]
    if data.get("production_batches") is not None:
        st.session_state["production_batches"] = data["production_batches"]
    if data.get("batch_counter") is not None:
        st.session_state["batch_counter"] = data["batch_counter"]

    SAFETY_STOCK_LEVEL = 106  # Updated safety stock level
    SAFETY_STOCK_UNITS = 102  # Additional safety stock units to add to first day's pallets

    # Ensure safety stock batch 0 is present
    if "production_batches" not in st.session_state:
        st.session_state["production_batches"] = []
    batches = st.session_state["production_batches"]
    if not any(b.get("batch_no") == 0 for b in batches):
        st.session_state["production_batches"].insert(0, {
            "batch_no": 0,
            "date": "Safety Stock",
            "pallets": round(SAFETY_STOCK_LEVEL / 7815),
            "TPV(hl)": float(SAFETY_STOCK_LEVEL),
            "shelf_life": 28,
            "shelf_life_left": 28
        })

    # Create tabs for different functionalities
    tabs = st.tabs(["Production Data Upload", "Stock and Expiry Monitoring", "Production Reports"])

    with tabs[0]:  # Production Data Upload
        st.header("Upload Production Data")

        # Align production dates with demand forecast period
        sales_data = load_data()
        last_forecast = sales_data.get("last_forecast")
        if last_forecast:
            start_date = pd.to_datetime(last_forecast["start_date"])
            end_date = pd.to_datetime(last_forecast["end_date"])
            pred_dates = pd.date_range(start=start_date, end=end_date)
            pred_dates = [d.strftime("%Y-%m-%d") for d in pred_dates]
        else:
            pred_dates = []

        # Use session state to store production records and batches
        if "production_records" not in st.session_state or st.session_state["production_records"] is None:
            st.session_state["production_records"] = [
                {
                    "date": date_str,
                    "standard brew(s)": 4,
                    "comment": "",
                    "TPV(hl)": "",
                    "deviation": "",
                    "actual brews": "",
                    "cause": ""
                }
                for date_str in pred_dates
            ]
        if "production_batches" not in st.session_state:
            st.session_state["production_batches"] = []

        prod_records = st.session_state["production_records"]
        prod_batches = st.session_state["production_batches"]

        # Prevent filling a later date before all prior dates are filled
        available_dates = [rec["date"] for rec in prod_records]
        next_idx = 0
        for i, rec in enumerate(prod_records):
            if rec["actual brews"] == "" or rec["actual brews"] is None:
                next_idx = i
                break
        else:
            next_idx = len(prod_records) - 1

        # Only allow entry for the next unfilled date unless in edit mode
        if "edit_mode" not in st.session_state:
            st.session_state["edit_mode"] = False

        if st.session_state["edit_mode"]:
            # Edit mode: allow selection of any date
            selected_date = st.selectbox("Select Date to Edit Production Data", available_dates, index=next_idx)
            rec_idx = next((i for i, rec in enumerate(prod_records) if rec["date"] == selected_date), None)
            rec = prod_records[rec_idx]

            st.markdown(f"**Editing production data for date:** `{selected_date}`")
        else:
            # Normal mode: show next unfilled date
            if not available_dates:
                st.warning("No available dates for production data entry.")
                return

            selected_date = available_dates[next_idx]
            rec_idx = next((i for i, rec in enumerate(prod_records) if rec["date"] == selected_date), None)
            rec = prod_records[rec_idx]

            st.markdown(f"**Entering production data for date:** `{selected_date}`")

        # Prompt for actual brews (allow decimals), production volume, comment
        actual_brews = st.number_input(
            "Actual Brews", min_value=0.0, value=round(float(rec["actual brews"]), 2) if str(rec["actual brews"]).replace('.','',1).isdigit() else 0.0,
            step=0.01, format="%.2f", key="prod_actual_brews"
        )
        comment = st.text_area("Comment on daily production activities", value=rec["comment"], key="prod_comment", height=120)
        # Updated cause options with codes
        cause_options = [
            ("mechanical failure (M/A)", "M/A"),
            ("electrical fault (E/F)", "E/F"),
            ("defects (D)", "D"),
            ("unscheduled maintenance (U/M)", "U/M"),
        ]
        selected_causes = []
        st.markdown("**Causes (select all that apply):**")
        for label, code in cause_options:
            if st.checkbox(label, value=(code in rec["cause"].split(", ")), key=f"cause_{code}_{selected_date}"):
                selected_causes.append(code)
        # If no cause is checked, set to 'n/a' in small letters
        cause_str = ", ".join(selected_causes) if selected_causes else "n/a"

        # Place Save and Edit buttons in line, just after causes
        col_save, col_edit = st.columns([2, 1])
        with col_save:
            save_btn_placeholder = st.empty()  # Placeholder for Save button
        with col_edit:
            edit_mode = st.checkbox("Edit previous entries", key="edit_mode_checkbox")

        # If edit mode, allow user to pick any date
        if edit_mode:
            selected_date = st.selectbox("Select Date to Edit Production Data", available_dates, index=rec_idx)
            rec_idx = next((i for i, rec in enumerate(prod_records) if rec["date"] == selected_date), None)
            rec = prod_records[rec_idx]
            st.markdown(f"**Editing production data for date:** `{selected_date}`")

        # Calculate pallets for the current entry
        tpv = round(actual_brews * 186225, 2) if actual_brews else ""  # Changed from 450000 to 186225
        # Add safety stock units to the first day's pallets
        is_first_day = rec_idx == 0
        pallets = int((tpv / 7815)) if tpv else 0
        if is_first_day:
            pallets += SAFETY_STOCK_UNITS

        # Save logic
        if save_btn_placeholder.button("Save Production Data", key="save_prod_data"):
            if not edit_mode:
                missing_prior = []
                for i in range(rec_idx):
                    if prod_records[i]["actual brews"] == "" or prod_records[i]["actual brews"] is None:
                        missing_prior.append(prod_records[i]["date"])
                if missing_prior:
                    st.warning(f"You must fill production data for all prior dates before entering data for {selected_date}. Missing: {', '.join(missing_prior)}")
                    st.stop()
            rec["actual brews"] = round(actual_brews, 2)
            rec["comment"] = comment
            rec["cause"] = cause_str
            rec["TPV(hl)"] = round(tpv, 2) if tpv != "" else ""
            rec["deviation"] = round(4 - actual_brews, 2) if actual_brews != "" else ""
            rec["pallets"] = pallets
            prod_records[rec_idx] = rec
            st.session_state["production_records"] = prod_records
                    
            # Batch handling
            if "batch_counter" not in st.session_state:
                st.session_state["batch_counter"] = 1  # Start from 1

            # --- Assign unique batch number and shelf life 28 to new batch --- 
            st.session_state["production_batches"].append({
                "batch_no": st.session_state["batch_counter"],
                "date": selected_date,
                "pallets": pallets,
                "TPV(hl)": tpv,
                "shelf_life": 28,
                "shelf_life_left": 28
            })
            st.session_state["batch_counter"] += 1

            # --- After adding the new batch, decrement shelf_life_left for all batches ---
            for b in st.session_state["production_batches"]:
                b["shelf_life_left"] = max(0, b["shelf_life_left"] - 1)

            # Ensure batch 1 always has shelf_life_left one less than batch 2 (if both exist)
            batches = st.session_state["production_batches"]
            if len(batches) >= 2:
                batches[0]["shelf_life_left"] = max(0, batches[1]["shelf_life_left"] - 1)

            # --- Immediately update remaining pallets for batch 1 by deducting sales on batch 1 date ---
            data = load_data()
            actual_sales = data.get("actual_sales", {})
            batch1 = next((b for b in batches if b.get("batch_no") == 1), None)
            if batch1:
                batch1_date = batch1.get("date")
                batch1_pallets = batch1.get("pallets", 0)
                batch1_sales = int(actual_sales.get(batch1_date, 0)) if batch1_date in actual_sales and str(actual_sales.get(batch1_date, "")).isdigit() else 0
                batch1_remain = batch1_pallets - batch1_sales
                if batch1_remain < 0:
                    batch1_remain = 0
                # Update batch1 pallets to remaining pallets
                batch1["pallets"] = batch1_remain
                # Update session state to reflect change
                st.session_state["production_batches"][batches.index(batch1)] = batch1

            # Save production data and batches immediately to persistent storage
            data = load_data()
            data["production_records"] = st.session_state["production_records"]
            data["production_batches"] = st.session_state["production_batches"]
            data["batch_counter"] = st.session_state["batch_counter"]
            data["edit_mode"] = st.session_state.get("edit_mode", False)
            save_data(data)

            st.success(f"Production data for {selected_date} saved.")
                
            # After saving, clear out the previous day's input for new entry (if not in edit mode)
            if not edit_mode:
                st.session_state.pop("prod_actual_brews", None)
                st.session_state.pop("prod_comment", None)
                for _, code in cause_options:
                    st.session_state.pop(f"cause_{code}_{selected_date}", None)
                if rec_idx + 1 < len(prod_records):
                    next_date = available_dates[rec_idx + 1]
                    st.info(f"Ready to enter production data for date: `{next_date}`")
                    st.rerun()
                else:
                    st.info("All production dates have been filled.")
                    # Always show the production table after the input section
        prod_df = pd.DataFrame(prod_records)
        for col in ["actual brews", "TPV(hl)", "deviation"]:
            if col in prod_df.columns:
                prod_df[col] = pd.to_numeric(prod_df[col], errors='coerce').round(2)
        column_config = {
            "date": st.column_config.Column("Date"),
            "actual brews": st.column_config.Column("Actual\nBrews", width="small"),
            "standard brew(s)": st.column_config.Column("Standard\nBrews", width="small"),
            "TPV(hl)": st.column_config.Column("TPV(hl)", width="small"),
            "deviation": st.column_config.Column("Deviation", width="small"),
            "comment": st.column_config.Column("Comment"),
            "cause": st.column_config.Column("Cause"),
            "pallets": st.column_config.Column("Pallets", width="small"),
        }
        st.dataframe(
            prod_df,
            column_config=column_config,
            use_container_width=True
        )

    with tabs[1]:  # Stock and Expiry Monitoring
        st.header("Monitor Stock and Expiry Dates")
        # SAFETY_STOCK_UNITS should be batch 0 value
        SAFETY_STOCK_UNITS = st.session_state["production_batches"][0]["pallets"] if st.session_state["production_batches"] else 0
        #st.info(f"System-wide Safety Stock Level: {SAFETY_STOCK_UNITS} units")

        # Get total sales up to date
        data = load_data()
        actual_sales = data.get("actual_sales", {})
        total_sales = sum([int(v) for v in actual_sales.values() if str(v).isdigit()])

        # Update pallets calculation for all batches (including batch 0)
        for b in st.session_state["production_batches"]:
            tpv = b.get("TPV(hl)", 0)
            b["pallets"] = int(round(float(tpv) / 7815)) if tpv else 0

        # --- Batch 0 shelf life decrement logic ---
        # Batch 0 shelf life decreases by 1 for each batch > 0 entered
        batch_0 = st.session_state["production_batches"][0]
        num_batches = len([b for b in st.session_state["production_batches"] if b.get("batch_no", 0) > 0])
        batch_0["shelf_life_left"] = max(28 - num_batches, 0)

        # --- System-wide change: Merge batch 0 (safety stock) into batch 1 ---
        # Only do this if there is at least batch 1
        batches = st.session_state["production_batches"]
        # Ensure each batch has a unique batch_no and one batch per date
        # Assign shelf life of 28 to new batch, decrement shelf life of prior batches by 1 on each new entry
        # This logic should be in the Production Data Upload save logic:
        # (Below is the correct place to update batch shelf lives and assign unique batch numbers)

        # Merge logic for batch 0 and batch 1
        if len(batches) > 1:
            # Merge batch 0 and batch 1
            batch0 = batches[0]
            batch1 = batches[1]
            # Ensure TPV(hl) values are numeric
            try:
                tpv0 = float(batch0.get("TPV(hl)", 0))
            except Exception:
                tpv0 = 0
            try:
                tpv1 = float(batch1.get("TPV(hl)", 0))
            except Exception:
                tpv1 = 0
            merged_pallets = batch0.get("pallets", 0) + batch1.get("pallets", 0)
            merged_tpv = tpv0 + tpv1
            # Create new batch 1 with merged values and shelf life 28
            merged_batch = {
                "batch_no": 1,
                "date": batch1.get("date", ""),
                "pallets": merged_pallets,
                "TPV(hl)": merged_tpv,
                "shelf_life": 28,
                "shelf_life_left": 28
            }
            st.session_state["production_batches"] = [merged_batch] + batches[2:]
            batches_fifo = st.session_state["production_batches"]
        else:
            batches_fifo = st.session_state["production_batches"]

    # Use the same dates as the production upload section (prediction period)
    prod_dates = sorted([rec["date"] for rec in st.session_state.get("production_records", [])])
    batch_table = []

    # --- Batch Pallet & Shelf-life Tracking ---
    batch_table = []
    prod_dates = sorted([rec["date"] for rec in st.session_state.get("production_records", [])])
    batches_fifo = st.session_state["production_batches"]

    # Synchronize batch shelf lives
    batches_fifo = sync_batch_shelf_lives(batches_fifo)

    # FIFO: For batch 1, remaining = pallets - sales. For batch 2 and after, remaining = pallets + prior batch's remaining - sales.
    prev_remain = None  # Track previous batch's remaining pallets
    prev_batch_no = None  # Track previous batch number

    last_numeric_idx = None
    remain_vals = []

    # First, calculate all remain_vals as before
    for idx, prod_date in enumerate(prod_dates):
        sales_val = int(actual_sales[prod_date]) if prod_date in actual_sales and str(actual_sales[prod_date]).isdigit() else 0
        batch_found = False

        for b in batches_fifo:
            if b.get("date") == prod_date:
                batch_no = b.get("batch_no")
                pallets_val = b.get("pallets", 0)

                if batch_no == 1:
                    remain_val = 102 - sales_val
                else:
                    remain_val = pallets_val + (prev_remain if prev_remain is not None else 0) - sales_val

                if remain_val < 0:
                    remain_val = 0

                remain_vals.append(remain_val)
                prev_remain = remain_val
                prev_batch_no = batch_no
                batch_found = True
                break

        if not batch_found:
            remain_vals.append(None)

    # Find the last index with a nonzero remain_val (the last batch with stock)
    last_numeric_idx = None
    for i in reversed(range(len(remain_vals))):
        if remain_vals[i] is not None and remain_vals[i] > 0:
            last_numeric_idx = i
            break
    if last_numeric_idx is None:
        # If all are zero or None, show the last as 0
        last_numeric_idx = len(remain_vals) - 1

    # Now build the batch_table with dashes for prior batches
    for idx, prod_date in enumerate(prod_dates):
        batch_found = False
        for b in batches_fifo:
            if b.get("date") == prod_date:
                batch_no = b.get("batch_no")
                remain_val = remain_vals[idx]
                # Only the last batch with stock shows a number, all prior show "-"
                if idx < last_numeric_idx:
                    display_remain = "-"  # Show dash for prior batches
                elif idx == last_numeric_idx:
                    display_remain = remain_val  # Only last batch with stock shows the number
                else:
                    display_remain = remain_val  # For future/unfilled batches, show the calculated value
                batch_table.append({
                    "Date": prod_date,
                    "Batch": batch_no,
                    "Shelf-life (days)": b["shelf_life_left"],
                    "Remaining Pallets": display_remain
                })
                batch_found = True
                break
        if not batch_found:
            continue

    # If no production dates yet, show initial state for all batches and empty date
    if not batch_table:
        batches_fifo = sync_batch_shelf_lives(batches_fifo)  # Sync before initial display
        for b in batches_fifo:
            batch_table.append({
                "Date": "",
                "Batch": b["batch_no"],
                "Shelf-life (days)": b["shelf_life_left"],
                "Remaining Pallets": b["pallets"]
            })

    batch_df = pd.DataFrame(batch_table)
    st.subheader("Batch Pallet & Shelf-life Tracking")
    st.dataframe(batch_df, use_container_width=True)

    # --- UNITS TO BE PRODUCED SUGGESTION ---
    # Get units to be sold from sales dashboard (reuse logic)
    data = load_data()
    pred_items = sorted(data.get("predictions", {}).items())
    actual_sales = data.get("actual_sales", {})
    total_predicted = sum([pred for _, pred in pred_items])

    # Current Stock Level: last displayed value in Remaining Pallets column
    if not batch_df.empty and last_numeric_idx is not None and 0 <= last_numeric_idx < len(batch_df):
        current_stock_display = batch_df["Remaining Pallets"].iloc[last_numeric_idx]
    else:
        current_stock_display = 0

    # --- Count number of predictions made in the system ---
    num_predictions = len(pred_items)

    # --- Count number of dashes in the batch status table ---
    num_dashes = batch_df["Remaining Pallets"].apply(lambda x: x == "-").sum() if not batch_df.empty else 0

    # --- Calculate remaining days ---
    remaining_days = max(num_predictions - (num_dashes + 1), 1)  # Always at least 1 to avoid division by zero

    # --- UNITS TO BE PRODUCED calculation (updated as per instruction) ---
    units_to_be_produced = total_predicted + 102 - (current_stock_display if isinstance(current_stock_display, (int, float)) else 0)

    # --- Calculate quantity to be produced ---
    quantity_to_be_produced = int(units_to_be_produced / remaining_days) if remaining_days > 0 else units_to_be_produced

    # --- Attach quantity_to_be_produced to the next production date ---
    sales_data = load_data()
    last_forecast = sales_data.get("last_forecast")
    pred_dates = []
    if last_forecast:
        start_date = pd.to_datetime(last_forecast["start_date"])
        end_date = pd.to_datetime(last_forecast["end_date"])
        pred_dates = pd.date_range(start=start_date, end=end_date)
        pred_dates = [d.strftime("%Y-%m-%d") for d in pred_dates]

    prod_records = st.session_state.get("production_records", [])
    next_prod_idx = 0
    for i, rec in enumerate(prod_records):
        if rec["actual brews"] == "" or rec["actual brews"] is None:
            next_prod_idx = i
            break
    else:
        next_prod_idx = len(prod_records)

    if next_prod_idx < len(pred_dates):
        next_prod_date = pred_dates[next_prod_idx]
    else:
        next_prod_date = None

    # --- Store the quantity_to_be_produced value for the next production date in session state as a mapping ---
    if "quantity_to_be_produced_map" not in st.session_state:
        st.session_state["quantity_to_be_produced_map"] = {}
    qt_map = st.session_state["quantity_to_be_produced_map"]

    # Always update the mapping for the next production date
    if next_prod_date:
        qt_map[next_prod_date] = quantity_to_be_produced
        st.session_state["quantity_to_be_produced_map"] = qt_map

    approval_status_list = st.session_state.get("approval_status_history", [])
    current_approval_status = approval_status_list[len(approval_status_list)-1] if approval_status_list and len(approval_status_list) == (num_dashes + 1) else "-"
    status_emoji = "✅" if current_approval_status == "approved" else "-"
    if quantity_to_be_produced < 0:
        st.success("There is enough stock. No production required.")
    else:
        st.info(f"Quantity to be Produced for {next_prod_date}: {quantity_to_be_produced} {status_emoji}")

    # After st.dataframe(batch_df, ...) in this section, store the last displayed numeric value for use in reports
    last_displayed_stock_level = None
    if not batch_df.empty:
        remaining_pallets_col = batch_df["Remaining Pallets"]
        numeric_vals = [v for v in remaining_pallets_col if isinstance(v, (int, float)) and not pd.isnull(v)]
        if numeric_vals:
            last_displayed_stock_level = numeric_vals[-1]
    st.session_state["current_stock_level_displayed"] = last_displayed_stock_level

def management_department():
    # Add background image and welcome message
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('traditional20%25flavour.jpg');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.title("WELCOME TO THE MANAGEMENT DEPARTMENT DASHBOARD")

    data = load_data()
    if data.get("management_reports") is not None:
        st.session_state["management_reports"] = data["management_reports"]

    tabs = st.tabs(["Approval Requests", "Sales Analysis", "Production Analysis", "System Overview"])

    # --- Approval Requests Tab ---
    with tabs[0]:
        st.header("Pending Approval Requests")
        sales_data = load_data()
        last_forecast = sales_data.get("last_forecast")
        pred_dates = []
        if last_forecast:
            start_date = pd.to_datetime(last_forecast["start_date"])
            end_date = pd.to_datetime(last_forecast["end_date"])
            pred_dates = pd.date_range(start=start_date, end=end_date)
            pred_dates = [d.strftime("%Y-%m-%d") for d in pred_dates]

        qt_map = st.session_state.get("quantity_to_be_produced_map", {})
        approval_status_list = st.session_state.get("approval_status_history", [])

        approval_table = []
        approval_dates = []
        for idx in range(1, len(pred_dates)):
            date = pred_dates[idx]
            qty = qt_map.get(date, "")
            status = approval_status_list[idx-1] if idx-1 < len(approval_status_list) else "-"
            status_display = "✅" if status == "approved" else "-"
            approval_table.append({
                "Date": date,
                "Quantity to be Produced": qty,
                "Status": status_display
            })
            approval_dates.append(date)

        approval_df = pd.DataFrame(approval_table)
        st.subheader("Pending Approval and Requests")

        # --- Manager Approval/Adjustment Section ---
        # Only allow approval/adjustment for the first pending (dash) entry
        first_pending_idx = next((i for i, row in enumerate(approval_table) if row["Status"] == "-"), None)
        rerun_triggered = False
        if first_pending_idx is not None:
            pending_entry = approval_table[first_pending_idx]
            st.markdown(f"### Review Production Target for {pending_entry['Date']}")
            current_qty = pending_entry["Quantity to be Produced"]
            if current_qty == "" or current_qty is None:
                st.info("No suggested quantity to be produced yet for this date.")
            else:
                try:
                    qty_val = int(current_qty)
                except Exception:
                    qty_val = 0
                new_qty = st.number_input(
                    "Adjust Quantity to be Produced (if needed)",
                    value=qty_val,
                    min_value=0,
                    step=1,
                    key="manager_adjust_qty"
                )
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Approve", key="approve_btn"):
                        if len(approval_status_list) < first_pending_idx + 1:
                            approval_status_list += ["-"] * (first_pending_idx + 1 - len(approval_status_list))
                        approval_status_list[first_pending_idx] = "approved"
                        qt_map[pending_entry["Date"]] = new_qty
                        st.session_state["approval_status_history"] = approval_status_list
                        st.session_state["quantity_to_be_produced_map"] = qt_map
                        rerun_triggered = True
                        st.success(f"Production target for {pending_entry['Date']} approved.")
                with col2:
                    if st.button("Reject", key="reject_btn"):
                        if len(approval_status_list) < first_pending_idx + 1:
                            approval_status_list += ["-"] * (first_pending_idx + 1 - len(approval_status_list))
                        approval_status_list[first_pending_idx] = "rejected"
                        st.session_state["approval_status_history"] = approval_status_list
                        rerun_triggered = True
                        st.warning(f"Production target for {pending_entry['Date']} rejected.")
        else:
            st.info("All production targets in the prediction period have been reviewed.")

        # Always show the updated table after any approval/rejection
        approval_table = []
        for idx in range(1, len(pred_dates)):
            date = pred_dates[idx]
            qty = qt_map.get(date, "")
            status = approval_status_list[idx-1] if idx-1 < len(approval_status_list) else "-"
            status_display = "✅" if status == "approved" else "-"
            approval_table.append({
                "Date": date,
                "Quantity to be Produced": qty,
                "Status": status_display
            })
        approval_df = pd.DataFrame(approval_table)
        st.dataframe(approval_df, use_container_width=True)

        if rerun_triggered:
            # Use st.rerun() for Streamlit >=1.25, else fallback to st.experimental_rerun()
            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()

    # --- Sales Analysis Tab ---
    with tabs[1]:
        st.header("Sales Performance Analysis")
        if last_forecast:
            analysis_start_date = pd.to_datetime(last_forecast["start_date"])
            analysis_end_date = pd.to_datetime(last_forecast["end_date"])
        else:
            analysis_start_date = datetime.now() - timedelta(days=7)
            analysis_end_date = datetime.now()

        # Get actual sales from sales upload section (Sales Department)
        sales_data = load_data()
        # Build actual sales DataFrame from the "Sales" column in the sales upload section
        # This is the same as the "actual_sales" dict in persistent storage
        actual_sales_dict = sales_data.get("actual_sales", {})
        actual_sales_list = []
        for date_str, sales_val in actual_sales_dict.items():
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d")
            except Exception:
                continue
            if analysis_start_date <= date <= analysis_end_date:
                try:
                    sales_val_int = int(sales_val)
                except Exception:
                    sales_val_int = None
                actual_sales_list.append((date, sales_val_int))

        # Get predicted sales from the predictions dict
        predicted_sales_dict = sales_data.get("predictions", {})
        predicted_sales_list = []
        for date_str, pred_val in predicted_sales_dict.items():
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d")
            except Exception:
                continue
            if analysis_start_date <= date <= analysis_end_date:
                predicted_sales_list.append((date, pred_val))

        # Build DataFrames
        df_actual = pd.DataFrame(actual_sales_list, columns=["Date", "Actual Sales"])
        df_pred = pd.DataFrame(predicted_sales_list, columns=["Date", "Predicted Sales"])

        # Merge on Date for plotting
        merged_df = pd.merge(df_actual, df_pred, on="Date", how="outer").sort_values("Date")

        # Plot: both actual sales and predicted sales as line graphs
        plt.figure(figsize=(12, 6))
        plt.plot(merged_df["Date"], merged_df["Actual Sales"], '-', color='blue', marker='o', label='Actual Sales')
        plt.plot(merged_df["Date"], merged_df["Predicted Sales"], '-', color='orange', marker='o', label='Predicted Sales')
        plt.title("Sales vs Date")
        plt.xlabel("Date")
        plt.ylabel("Sales")
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
        st.pyplot(plt)

        # Calculate metrics
        merged_df_clean = merged_df.dropna()

        if not merged_df_clean.empty:
            error = abs(merged_df_clean['Actual Sales'] - merged_df_clean['Predicted Sales'])
            mae = error.mean()
            mape = (error / merged_df_clean['Predicted Sales']).mean() * 100

            st.subheader("Forecast Accuracy Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Absolute Error", f"{mae:.2f} units")
            with col2:
                st.metric("Mean Absolute Percentage Error", f"{mape:.2f}%")

            st.subheader("Forecast Accuracy by Day")
            accuracy_df = merged_df_clean.copy()
            accuracy_df['Error'] = error
            accuracy_df['Error_Percent'] = (error / accuracy_df['Predicted Sales']) * 100
            st.dataframe(accuracy_df[['Date', 'Actual Sales', 'Predicted Sales', 'Error', 'Error_Percent']])

            report_data = {
                "Start Date": analysis_start_date.strftime("%Y-%m-%d"),
                "End Date": analysis_end_date.strftime("%Y-%m-%d"),
                "Total Actual Sales": merged_df['Actual Sales'].sum(),
                "Total Predicted Sales": merged_df['Predicted Sales'].sum(),
                "Mean Absolute Error": f"{mae:.2f} units",
                "Mean Absolute Percentage Error": f"{mape:.2f}%",
                "Average Daily Sales": merged_df['Actual Sales'].mean(),
                "Maximum Daily Sales": merged_df['Actual Sales'].max()
            }
            download_pdf_button(report_data, "Sales Analysis Report")
        else:
            st.warning("No matching dates between actual sales and predictions.")
        st.subheader("Sales Performance Data")
        st.dataframe(merged_df)

    # --- Production Analysis Tab ---
    with tabs[2]:  # Production Analysis
        st.header("Production Analysis")

        # Use prediction period dates
        if last_forecast:
            prod_start_date = pd.to_datetime(last_forecast["start_date"])
            prod_end_date = pd.to_datetime(last_forecast["end_date"])
        else:
            prod_start_date = datetime.now() - timedelta(days=7)
            prod_end_date = datetime.now()

        # Import filled table from production data upload (from session state)
        prod_records = st.session_state.get("production_records", [])
        prod_df = pd.DataFrame(prod_records)
        if not prod_df.empty:
            prod_df['date'] = pd.to_datetime(prod_df['date'])
            prod_df = prod_df.sort_values('date')
            st.subheader("Production Data Table")
            st.dataframe(prod_df, use_container_width=True)

            # Correction: Plot TPV(hl) vs date using values from the table
            if "TPV(hl)" in prod_df.columns and "date" in prod_df.columns:
                tpv_df = prod_df[["date", "TPV(hl)"]].copy()
                tpv_df = tpv_df.dropna(subset=["TPV(hl)"])
                tpv_df["TPV(hl)"] = pd.to_numeric(tpv_df["TPV(hl)"], errors="coerce")
                tpv_df = tpv_df.dropna(subset=["TPV(hl)"])
                if not tpv_df.empty:
                    plt.figure(figsize=(12, 6))
                    plt.plot(tpv_df["date"], tpv_df["TPV(hl)"], marker='o', label='TPV(hl)')
                    plt.title("Production Volume (TPV(hl)) Over Time")
                    plt.xlabel("Date")
                    plt.ylabel("TPV(hl)")
                    plt.grid(True)
                    plt.legend()
                    plt.xticks(rotation=45)
                    fig = plt.gcf()
                    st.pyplot(fig)
                else:
                    st.info("No valid production volume data to plot.")
            else:
                st.info("No valid production volume data to plot.")
        else:
            st.info("No production data available.")

    # --- System Overview Tab ---
    with tabs[3]:  # System Overview
        st.header("System Overview")

        # Blend in the report details section decoratively
        st.markdown("""
        <div style="background-color:#f0f2f6; border-radius:10px; padding:20px; margin-bottom:20px; border:1px solid #e0e0e0;">
            <h4 style="color:#2c3e50;">📋 Report Details</h4>
        """, unsafe_allow_html=True)

        # Find the last approved Quantity to be Produced
        last_approved = None
        for entry in reversed(approval_table):
            if entry["Status"] == "✅":
                last_approved = entry
                break

        # Get current stock level (system-wide, from production dashboard logic)
        # Use the last displayed stock level from the production department
        current_stock = st.session_state.get("current_stock_level_displayed", "Not Available")

        if last_approved:
            # Show details for last approved Quantity to be Produced
            date = last_approved['Date']
            qty = last_approved['Quantity to be Produced']
            # If you have per-date stock levels, you can fetch the stock for that date, else use current_stock
            st.markdown(f"""
            <b>Date:</b> {date}<br>
            <b>Quantity to be Produced:</b> {qty}<br>
            <b>Current Stock Level:</b> {current_stock}
            """, unsafe_allow_html=True)
        else:
            st.markdown("<i>No approved production target available.</i>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Show only Current Stock and Pending Approvals (Expiring Soon removed)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Stock", f"{current_stock} units")
        with col2:
            st.metric("Pending Approvals", "0")

        # User activity and recent approvals remain unchanged
        st.subheader("User Activity")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Production Activity**")
            # Count the number of non-empty entries in the 'pallets' column of the production data table
            prod_records = st.session_state.get("production_records", [])
            num_pallets_entries = sum(1 for rec in prod_records if rec.get("pallets") not in (None, "", 0))
            st.info(f"Total production records (pallets entries): {num_pallets_entries}")
        with col2:
            st.write("**Sales Activity**")
            sales_data = get_sales_data()
            # Count the number of entries in the actual_sales dictionary
            num_sales_records = len(data.get("actual_sales", {}))
            if sales_data:
                st.info(f"Total sales records: {num_sales_records}")
            else:
                st.warning("No sales data available.")

        st.subheader("Recent Approval Activity")
        # Show the most recently approved quantity to be produced
        recent_approved_qty = None
        for entry in reversed(approval_table):
            if entry["Status"] == "✅":
                recent_approved_qty = entry.get("Quantity to be Produced")
                break
        if recent_approved_qty is not None:
            st.info(f"Recently Approved Quantity to be Produced: {recent_approved_qty}")
        else:
            st.info("No recent approval activity.")


def get_current_stock_and_batch_status(prod_batches, total_sales):
    """
    Returns:
        current_stock: int, current units in stock
        batch_status: list of dicts with keys: batch_no, date, shelf_life_left, remaining_pallets
    """
    # Build a list of all batches in FIFO order
    batches = sorted(prod_batches, key=lambda b: b.get("batch_no", 0))
    # Each batch: {'batch_no', 'date', 'pallets', 'shelf_life_left'}
    batch_units = []
    for b in batches:
        # Each pallet is 1 unit (for calculation), or use b["pallets"] as units
        batch_units.append({
            "batch_no": b.get("batch_no", 0),
            "date": b.get("date", ""),
            "shelf_life_left": b.get("shelf_life_left", 0),
            "pallets": b.get("pallets", 0),
            "remaining_pallets": b.get("pallets", 0)  # will be updated below
        })

    # FIFO deduction of sales from batches
    sales_left = total_sales
    for b in batch_units:
        if sales_left <= 0:
            break
        if b["remaining_pallets"] > 0:
            if b["remaining_pallets"] >= sales_left:
                b["remaining_pallets"] -= sales_left
                sales_left = 0
            else:
                sales_left -= b["remaining_pallets"]
                b["remaining_pallets"] = 0

    # Current stock is sum of remaining pallets in all batches
    current_stock = sum(b["remaining_pallets"] for b in batch_units)
    return current_stock, batch_units

def logout():
    """Logout the user while preserving session data and ensuring all records are saved."""
    # Save all persistent data before logging out
    data = load_data()
    # Update persistent data with current session state if available
    if "production_records" in st.session_state:
        data["production_records"] = st.session_state["production_records"]
    if "production_batches" in st.session_state:
        data["production_batches"] = st.session_state["production_batches"]
    if "batch_counter" in st.session_state:
        data["batch_counter"] = st.session_state["batch_counter"]
    if "management_reports" in st.session_state:
        data["management_reports"] = st.session_state["management_reports"]
    if "approval_status_history" in st.session_state:
        data["approval_status_history"] = st.session_state["approval_status_history"]
    if "quantity_to_be_produced_map" in st.session_state:
        data["quantity_to_be_produced_map"] = st.session_state["quantity_to_be_produced_map"]
    if "current_stock_level_displayed" in st.session_state:
        data["current_stock_level_displayed"] = st.session_state["current_stock_level_displayed"]
    # Save any other relevant session state keys as needed

    save_data(data)

    st.session_state['logged_in'] = False
    st.session_state['username'] = None
    st.session_state['department'] = None
    st.session_state['current_page'] = "home"
    st.success("You have been logged out successfully!")

def main():
    st.sidebar.title("Menu")

    # Load models at the start of the app
    model_low, model_high = load_models()

    # Check if user is logged in
    if st.session_state['logged_in']:
        st.sidebar.header(f"Welcome, {st.session_state['username']}!")
        if st.sidebar.button("Logout"):
            logout()

        # Show department-specific navigation
        if st.session_state['department'] == "sales":
            sales_department(model_low, model_high)
        elif st.session_state['department'] == "production":
            production_department()
        elif st.session_state['department'] == "management":
            management_department()
        else:
            # Default department
            st.write(
                f"Welcome {st.session_state['username']}! Your department ({st.session_state['department']}) is not recognized.")
    else:
        # Not logged in, show login or signup options
        menu = st.sidebar.selectbox("Action", ["Home", "Login", "SignUp"])

        if menu == "Home":
            # Add background image for home page only (handle spaces in filename)
            st.markdown(
                """
                <style>
                .stApp {
                    background-image: url('traditional20%25flavour.jpg');
                    background-size: cover;
                    background-repeat: no-repeat;
                    background-attachment: fixed;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
            st.text("⬅️⬅️⬅️Click side bar to login...")
            st.title("Chibuku Beverages Sales Prediction System")

            st.subheader("Home")
            st.text("What is demand forecasting?🤔")

            # Using st.text_area to display the second piece of text in a scrollable text box
            demand_forecasting_description = (
                "Demand forecasting is the process of predicting future customer demand for products or services "
                "over a specific time frame. By analyzing historical sales data and market trends, businesses can make "
                "informed decisions about inventory management, production planning, and resource allocation. "
                "Accurate demand forecasts help reduce costs, avoid stocks outage, and enhance overall operational efficiency, "
                "enabling companies to meet customer needs effectively."
            )
            # Use st.markdown to create a white text box
            st.markdown(
                f"""
                <div style="background-color: grey; padding: 10px; border-radius: 5px; border: 1px solid #ddd;">
                    <p>{demand_forecasting_description}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Custom CSS to style the frames with rounded corners
            st.markdown("""
                <style>
                .image-frame {
                    border: 1px solid #ddd; 
                    border-radius: 10px; 
                    padding: 10px; 
                    background-color: white; 
                    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2); 
                    text-align: center;
                }
                </style>
            """, unsafe_allow_html=True)

            # Create three columns for content
            col1, col2, col3 = st.columns(3)

            # Display text in each column
            with col1:
                st.markdown(
                    """
                    ### 📊 Dashboard
                    Visualize your beverage sales trends with an intuitive dashboard. Monitor key metrics to make informed, data-driven decisions and enhance your revenue.
                    """
                )

            with col2:
                st.markdown(
                    """
                    ### 📈 Predictions
                    Utilize advanced predictive analytics to forecast future sales accurately. Our tool analyzes historical data to help you anticipate demand and stay competitive.
                    """
                )

            with col3:
                st.markdown(
                    """
                    ### 📝 Reports
                    Generate tailored reports that fit your business needs. Track specific metrics and timeframes to gain valuable insights into your sales performance and identify growth opportunities.
                    """
                )

            # Department information
            st.header("Department Portals")

            dept_col1, dept_col2, dept_col3 = st.columns(3)

            with dept_col1:
                st.markdown(
                    """
                    ### 🛒 Sales Department
                    - Forecast demand for specified periods
                    - Upload daily sales data
                    - Monitor stock levels
                    - Generate sales reports
                    """
                )

            with dept_col2:
                st.markdown(
                    """
                    ### 🏭 Production Department
                    - Upload daily production data
                    - Receive production adjustment recommendations
                    - Manage stock and expiry dates
                    - Generate production reports
                    """
                )

            with dept_col3:
                st.markdown(
                    """
                    ### 👨‍💼 Management Department
                    - Review and approve production recommendations
                    - Analyze sales performance
                    - Monitor production efficiency
                    - System overview and reporting
                    """
                )

            # Paths to the images
            image_path1 = "c.png"
            image_path2 = "b.webp"
            image_path3 = "c.png"

            # Image file 'E274C19F-E9DE-428E-A966-00DBDD781D1E-1200x480.jpeg' not found, skipping image display
            # st.image("E274C19F-E9DE-428E-A966-00DBDD781D1E-1200x480.jpeg",
            #          use_container_width=True)


        elif menu == "Login":
            login()

        elif menu == "SignUp":
            signup()


if __name__ == "__main__":
    main()

# To make your Streamlit app accessible over the internet as a website, follow these steps:

# 1. Deploy to Streamlit Community Cloud (Recommended for most users)
#    - Go to https://streamlit.io/cloud
#    - Sign in with your GitHub account.
#    - Push your project (including this .py file and any dependencies) to a public GitHub repository.
#    - Click "New app", select your repo and branch, and set the main file path (e.g., streamlit_beer_production_system.py).
#    - Click "Deploy". Your app will be live at a public URL.

# 2. Alternative: Deploy on your own server or cloud VM
#    - Use a cloud provider (AWS, Azure, GCP, DigitalOcean, etc.) to create a virtual machine.
#    - Install Python and Streamlit on the VM.
#    - Upload your project files to the VM.
#    - Run: `streamlit run streamlit_beer_production_system.py --server.port 80 --server.address 0.0.0.0`
#    - Open firewall for port 80. Your app will be accessible via the VM's public IP.

# 3. (Optional) Use a custom domain
#    - Register a domain and point it to your Streamlit Cloud app or your VM's public IP.

# 4. (Optional) For production, consider using HTTPS (SSL) and a reverse proxy (e.g., Nginx).

# For most users, Streamlit Community Cloud is the easiest and fastest way to share your app as a website.
