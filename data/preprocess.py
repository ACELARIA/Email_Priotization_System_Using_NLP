"""
data/preprocess.py
------------------
Dataset loading and text preprocessing utilities for the Email Priority
Classification System.

Supports:
- Synthetic dataset generation (~2000 labelled samples across 4 classes)
- Optional Enron dataset loading when the CSV is present on disk
- Full NLP preprocessing pipeline (HTML removal, tokenisation, stopword
  removal, lemmatisation via NLTK)
"""

import re
import os
import random
import logging

import nltk
import pandas as pd

# ---------------------------------------------------------------------------
# NLTK resource bootstrap – download silently if not present
# ---------------------------------------------------------------------------
_NLTK_RESOURCES = [
    ("tokenizers/punkt",          "punkt"),
    ("tokenizers/punkt_tab",      "punkt_tab"),
    ("corpora/stopwords",         "stopwords"),
    ("corpora/wordnet",           "wordnet"),
    ("corpora/omw-1.4",           "omw-1.4"),
]

for _resource_path, _resource_id in _NLTK_RESOURCES:
    try:
        nltk.data.find(_resource_path)
    except LookupError:
        nltk.download(_resource_id, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Fixed random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Synthetic email templates
# ---------------------------------------------------------------------------

# Each entry is a (subject, body) pair template.
_HIGH_PRIORITY_TEMPLATES = [
    (
        "URGENT: Production server is down",
        "The main production server crashed {minutes} minutes ago. All customer-facing services are affected. "
        "We need immediate action to restore operations. Please escalate to the on-call engineer right away.",
    ),
    (
        "Action required: Critical deadline today at {hour}pm",
        "This is a reminder that the deliverable for the {project} project is due today at {hour}pm. "
        "Failure to submit will result in contract penalties. Please respond immediately with your status.",
    ),
    (
        "URGENT: Security breach detected in {system}",
        "Our monitoring system has detected a potential security breach in {system}. "
        "Immediate investigation required. Do not delay — customer data may be at risk.",
    ),
    (
        "Critical bug causing data loss in production",
        "A critical bug has been identified in the {module} module that is causing data loss for active users. "
        "This needs to be fixed within the next {minutes} minutes. Assigning to senior dev team immediately.",
    ),
    (
        "Immediate response needed: Client escalation",
        "Our key client {client} has escalated an issue to executive level. "
        "They are threatening to terminate the contract unless we resolve the problem by end of day. "
        "Please coordinate with the account team immediately.",
    ),
    (
        "ALERT: Database connection pool exhausted",
        "Database connection pool for {service} has been exhausted. "
        "New requests are being rejected. Immediate intervention required to prevent full service outage.",
    ),
    (
        "Emergency: Build pipeline broken — release blocked",
        "The CI/CD pipeline has been broken since {time}. "
        "The upcoming {version} release is now blocked. All hands on deck to diagnose and fix this issue.",
    ),
    (
        "Action required: Compliance audit due TODAY",
        "The regulatory compliance audit submission is due today by 5pm. "
        "Several documents are still missing from the submission package. Please action immediately.",
    ),
    (
        "URGENT: Payment processing system failing",
        "The payment gateway is returning errors for {percent}% of transactions. "
        "Revenue impact is significant. Engineering team must prioritise this issue above all else.",
    ),
    (
        "Critical: Disk space on {server} at {percent}% — service will crash soon",
        "Disk utilisation on {server} has reached {percent}%. "
        "If this is not addressed within the hour, the service will crash and data may be lost.",
    ),
]

_MEDIUM_PRIORITY_TEMPLATES = [
    (
        "Meeting tomorrow at {hour}pm — {topic}",
        "Hi team, just a reminder that we have a meeting tomorrow at {hour}pm to discuss {topic}. "
        "Please review the agenda document beforehand and come prepared with updates.",
    ),
    (
        "Sprint review this Friday — please prepare your demos",
        "Our sprint review is scheduled for this Friday at 3pm. "
        "Each team member should prepare a brief demo of what they worked on during the sprint. "
        "Let me know if you have any questions.",
    ),
    (
        "Follow-up on yesterday's discussion about {topic}",
        "Following up on our conversation yesterday regarding {topic}. "
        "Could you please send me the relevant documents or a summary of the next steps by Thursday?",
    ),
    (
        "Project update: {project} — week {week} status",
        "Here is the weekly status update for the {project} project. "
        "Progress is on track for most milestones. There are a couple of minor blockers that need attention. "
        "Full report attached.",
    ),
    (
        "Request for review: {document} draft",
        "Hi, I have completed a first draft of the {document}. "
        "Could you please review it and provide feedback by end of the week? "
        "Your input would be greatly appreciated.",
    ),
    (
        "Proposal for Q{quarter} roadmap — your input needed",
        "I am putting together the Q{quarter} product roadmap and would love to get your perspective. "
        "Could we schedule a 30-minute call this week to discuss priorities?",
    ),
    (
        "Reminder: performance reviews due by {date}",
        "Just a reminder that manager performance reviews are due by {date}. "
        "Please ensure you have completed assessments for all direct reports in the HR portal.",
    ),
    (
        "Team lunch organised for {day} — RSVP needed",
        "We are organising a team lunch on {day} at {restaurant}. "
        "Please RSVP by Wednesday so we can confirm the booking. Hope to see everyone there!",
    ),
    (
        "New process for handling {process} — please read",
        "Starting from next Monday, we will be adopting a new process for {process}. "
        "Details are in the attached document. Please familiarise yourself before the changeover date.",
    ),
    (
        "Feedback requested on the {topic} presentation",
        "I presented the {topic} proposal to the leadership team earlier today. "
        "They had several questions and requested more detail on the financials. "
        "Could you help me refine the slides?",
    ),
]

_LOW_PRIORITY_TEMPLATES = [
    (
        "FYI: Company newsletter — {month} edition",
        "Read the {month} edition of our company newsletter! "
        "This month we highlight new hires, upcoming events, and a spotlight on the {team} team. "
        "No action required — just for your information.",
    ),
    (
        "Reminder: office hours next week with {name}",
        "Just a heads-up that {name} will be holding open office hours next week on {day}. "
        "Feel free to drop in if you have any questions. No appointment necessary.",
    ),
    (
        "Read when you get a chance: article on {topic}",
        "I came across this interesting article on {topic} and thought it might be relevant to our work. "
        "No rush — read it whenever you have a free moment.",
    ),
    (
        "Updated parking guidelines — effective next month",
        "Please note that the parking policy has been updated effective next month. "
        "The main changes relate to reserved spots and visitor parking. "
        "Read the full document whenever convenient.",
    ),
    (
        "Optional training webinar: {topic} — {date}",
        "There is an optional training webinar on {topic} scheduled for {date}. "
        "Attendance is entirely voluntary. Registration link is in the attachment.",
    ),
    (
        "Office kitchen — friendly reminder about cleanliness",
        "Just a friendly reminder to please keep the office kitchen tidy. "
        "Label your food in the fridge and clean up after yourself. Thank you for your co-operation!",
    ),
    (
        "FYI: IT maintenance scheduled for {date} — low impact",
        "Routine IT maintenance is scheduled for {date} between 11pm and 1am. "
        "There may be a brief interruption to internal services. No action required from your side.",
    ),
    (
        "Interesting read: {topic} best practices",
        "Sharing a useful resource on {topic} best practices. "
        "No urgency — just something I thought the team might find helpful when they have time.",
    ),
    (
        "Team lunch this Friday at {restaurant}",
        "We are having a casual team lunch this Friday at {restaurant}. "
        "Everyone is welcome — completely informal and optional. Hope to see you there!",
    ),
    (
        "Congratulations to {name} on their work anniversary!",
        "Please join me in congratulating {name} on {years} years with the company! "
        "It has been a pleasure working with them. No reply needed — just wanted to share the good news.",
    ),
]

_SPAM_TEMPLATES = [
    (
        "Congratulations! You have been selected for a special reward",
        "You are one of our lucky winners! Click the link below to claim your exclusive reward. "
        "This offer expires in 24 hours. Provide your details to receive your prize immediately.",
    ),
    (
        "CLAIM NOW: You won ${amount} in our monthly draw",
        "Your email address was randomly selected in our monthly prize draw. "
        "You have won ${amount}! To claim your prize, click here and verify your bank account details.",
    ),
    (
        "Limited time offer — {percent}% off everything, today only!",
        "Massive sale — {percent}% discount on all products for the next 24 hours only! "
        "Use code SAVE{percent} at checkout. Don't miss this incredible deal. Buy now!",
    ),
    (
        "Your account has been suspended — verify now",
        "We have detected unusual activity on your account. "
        "Your access has been temporarily suspended. Click the link to verify your identity and restore access. "
        "Failure to verify within 48 hours will result in permanent account closure.",
    ),
    (
        "Make ${amount} per week working from home — guaranteed",
        "Discover the secret to earning ${amount} a week from the comfort of your home! "
        "No experience required. Thousands of people are already doing this. Click here to get started.",
    ),
    (
        "FINAL NOTICE: Unclaimed inheritance of ${amount}",
        "We have been trying to reach you regarding an unclaimed inheritance of ${amount}. "
        "You are listed as the beneficiary. Contact us immediately to arrange the transfer.",
    ),
    (
        "Your parcel could not be delivered — action required",
        "We attempted to deliver your parcel but were unable to complete delivery. "
        "Pay the outstanding customs fee of ${amount} to release your package. Click here to pay.",
    ),
    (
        "Hot singles in your area are waiting to meet you",
        "Thousands of singles near you are looking to connect! "
        "Sign up for free today and start chatting. No credit card needed to get started.",
    ),
    (
        "You have been pre-approved for a ${amount} loan",
        "Great news! You have been pre-approved for a personal loan of up to ${amount}. "
        "No credit check required. Click here to get your funds transferred within 24 hours.",
    ),
    (
        "Exclusive invitation: Double your investment in 30 days",
        "Join our exclusive investment platform and double your money in 30 days — guaranteed! "
        "Our proprietary algorithm has a 100% success rate. Limited spots available. Sign up now.",
    ),
]

# Filler values for template placeholders
_FILL_VALUES = {
    "minutes":   ["5", "10", "15", "20", "30"],
    "hour":      ["2", "3", "4", "5", "9", "10", "11"],
    "project":   ["Phoenix", "Atlas", "Orion", "Titan", "Nexus", "Aurora"],
    "system":    ["authentication service", "payment gateway", "user database", "API layer"],
    "module":    ["billing", "reporting", "user management", "checkout", "analytics"],
    "client":    ["Acme Corp", "GlobalTech", "Pinnacle Inc", "SkyBridge", "Vertex Ltd"],
    "service":   ["orders-api", "auth-service", "data-pipeline", "notification-service"],
    "time":      ["9:15am", "10:30am", "11:00am", "2:45pm"],
    "version":   ["v2.4.0", "v3.1.0", "v1.8.5", "v4.0.0"],
    "server":    ["prod-db-01", "app-server-02", "data-store-03", "cache-01"],
    "percent":   ["85", "90", "92", "95", "98"],
    "topic":     ["data pipelines", "microservices", "cloud migration", "security hardening",
                  "API design", "DevOps", "machine learning", "agile workflows"],
    "document":  ["technical specification", "project proposal", "architecture diagram", "API docs"],
    "quarter":   ["1", "2", "3", "4"],
    "date":      ["Friday 15th", "Monday 18th", "Wednesday 20th", "Thursday 21st"],
    "day":       ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
    "restaurant":["The Bistro", "Noodle House", "The Garden Cafe", "Brick Lane Kitchen"],
    "process":   ["incident management", "code review", "deployment", "expense reporting"],
    "month":     ["January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"],
    "team":      ["engineering", "product", "design", "data science", "customer success"],
    "name":      ["Sarah", "James", "Priya", "Carlos", "Mei", "Alex", "Jordan"],
    "years":     ["1", "2", "3", "5", "7", "10"],
    "week":      ["1", "2", "3", "4", "5", "6", "7", "8"],
    "amount":    ["500", "1000", "2500", "5000", "10000", "50000"],
}


def _fill_template(template_str: str) -> str:
    """Replace all {placeholder} tokens in a template with random values."""
    result = template_str
    for key, values in _FILL_VALUES.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, random.choice(values))
    return result


def _generate_samples_for_class(
    templates: list,
    label: str,
    count: int,
) -> list:
    """
    Generate *count* synthetic email samples from the provided templates.

    Each sample is a dict with keys: subject, body, label.
    Templates are cycled and filled with random placeholder values.
    """
    samples = []
    for i in range(count):
        subject_tmpl, body_tmpl = templates[i % len(templates)]
        samples.append(
            {
                "subject": _fill_template(subject_tmpl),
                "body":    _fill_template(body_tmpl),
                "label":   label,
            }
        )
    return samples


def generate_synthetic_dataset(total_samples: int = 2000) -> pd.DataFrame:
    """
    Generate a synthetic email dataset with four priority categories.

    Parameters
    ----------
    total_samples : int
        Approximate total number of samples to generate. Divided equally
        across the four classes.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: subject, body, label, text.
        ``text`` is the concatenation of subject and body and is the primary
        feature column used for model training.
    """
    random.seed(RANDOM_SEED)
    per_class = total_samples // 4

    logger.info("Generating %d synthetic email samples (%d per class)...", total_samples, per_class)

    all_samples = (
        _generate_samples_for_class(_HIGH_PRIORITY_TEMPLATES,  "High Priority",   per_class)
        + _generate_samples_for_class(_MEDIUM_PRIORITY_TEMPLATES, "Medium Priority", per_class)
        + _generate_samples_for_class(_LOW_PRIORITY_TEMPLATES,  "Low Priority",    per_class)
        + _generate_samples_for_class(_SPAM_TEMPLATES,           "Spam",            per_class)
    )

    df = pd.DataFrame(all_samples)
    # Combined feature column used by the TF-IDF vectorizer
    df["text"] = df["subject"] + " " + df["body"]

    # Shuffle rows with a fixed seed
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    logger.info("Dataset generated: %d rows, classes: %s", len(df), df["label"].value_counts().to_dict())
    return df


# ---------------------------------------------------------------------------
# Text preprocessing pipeline
# ---------------------------------------------------------------------------

_stop_words = set(stopwords.words("english"))
_lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline.

    Steps
    -----
    1. Lowercase
    2. Remove HTML tags
    3. Remove special characters and digits, keeping only letters and spaces
    4. Tokenise using NLTK word_tokenize
    5. Remove stopwords
    6. Lemmatise each token

    Parameters
    ----------
    text : str
        Raw email text (subject + body combined, or any free text).

    Returns
    -------
    str
        Clean, space-joined token string ready for TF-IDF vectorisation.
    """
    if not isinstance(text, str):
        return ""

    # Step 1 – lowercase
    text = text.lower()

    # Step 2 – strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Step 3 – remove non-alpha characters (keep spaces)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Step 4 – tokenise
    tokens = word_tokenize(text)

    # Steps 5 & 6 – remove stopwords and lemmatise
    tokens = [
        _lemmatizer.lemmatize(token)
        for token in tokens
        if token not in _stop_words and len(token) > 1
    ]

    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(enron_csv_path: str | None = None) -> pd.DataFrame:
    """
    Load the email dataset.

    If *enron_csv_path* is provided and the file exists, it attempts to load
    the Enron dataset from that CSV (expects columns: subject, body, label).
    Falls back to the synthetic generator if loading fails or the path is not
    supplied.

    Parameters
    ----------
    enron_csv_path : str or None
        Optional file path to an Enron dataset CSV.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: subject, body, label, text.
    """
    if enron_csv_path and os.path.exists(enron_csv_path):
        try:
            logger.info("Loading Enron dataset from %s ...", enron_csv_path)
            df = pd.read_csv(enron_csv_path)
            required_cols = {"subject", "body", "label"}
            if not required_cols.issubset(df.columns):
                raise ValueError(f"CSV must contain columns: {required_cols}")
            df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")
            logger.info("Enron dataset loaded: %d rows.", len(df))
            return df
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load Enron dataset (%s). Falling back to synthetic data.", exc)

    # Default: synthetic dataset
    return generate_synthetic_dataset()
