
KEYWORD_SUGGESTIONS = {
    "harmonics": "To view harmonics, go to the Trend Graph tab and select 'Harmonics' from the dropdown.",
    "event list": "The Event List shows power quality events such as dips, swells, and inrush, with timestamps and channels.",
    "trend graph": "You can zoom, adjust period, and select measurement parameters in the Trend Graph panel.",
    "pq check": "PQ Check (Standards) can be found in the Statistics tab for EN50160 compliance evaluation.",
    "csv export": "Click the CSV icon on the toolbar to export trend or waveform data into Excel-readable format.",
    "report": "To create a report, click the report icon and configure trend, event, and statistics sections.",
    "pqdif": "PQDIF files can be saved using [Save as PQDIF] or opened via the [Open PQDIF File] option."
}

def check_proactive_suggestions(user_input: str):
    user_input = user_input.lower()
    for keyword, suggestion in KEYWORD_SUGGESTIONS.items():
        if keyword in user_input:
            return f"[Suggestion based on '{keyword}']:\n{suggestion}"
    return None
