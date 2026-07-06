"""
generate_dataset.py
Builds a labeled dataset of spam / ham (legitimate) email/SMS-style messages
using template-based synthesis with randomized slot-filling, so the resulting
corpus has realistic lexical variety and class-conditional word statistics
for a Naive Bayes model to actually learn from.
"""
import random
import csv

random.seed(42)

# ---------- SPAM building blocks ----------
spam_openers = [
    "CONGRATULATIONS", "URGENT NOTICE", "FINAL REMINDER", "ATTENTION",
    "Dear Winner", "Dear Valued Customer", "IMPORTANT ALERT", "ACT NOW",
    "You have been selected", "GREAT NEWS", "LIMITED TIME OFFER",
]
spam_hooks = [
    "you have won a ${amount} {prize}",
    "your account will be suspended unless you verify immediately",
    "you've been chosen to receive a free {prize}",
    "claim your ${amount} cash prize before it expires",
    "we noticed unusual activity on your account, verify now",
    "you are eligible for a {amount}% discount on {prize}",
    "your package could not be delivered, confirm your address",
    "your subscription payment of ${amount} has failed, update billing",
    "you qualify for a low interest loan of ${amount}",
    "your {prize} is ready to ship, confirm shipping details",
    "earn ${amount} per week working from home, no experience needed",
    "your PayPal account has been limited, restore access now",
]
spam_prizes = ["iPhone 16", "gift card", "cruise vacation", "lottery jackpot",
               "MacBook Pro", "Amazon voucher", "luxury watch", "free trial"]
spam_ctas = [
    "Click here to claim now: {link}",
    "Verify your account immediately at {link}",
    "Reply YES to redeem your reward",
    "Call {phone} within 24 hours or your offer expires",
    "Visit {link} and enter your details to confirm",
    "Download the attachment to complete verification",
    "Login now at {link} to secure your funds",
    "Send your bank details to {email} to receive payment",
]
spam_closers = [
    "Do not miss this exclusive opportunity!!!",
    "This offer expires in 24 hours, act fast!",
    "100% guaranteed, no risk, no obligation.",
    "Limited slots available, first come first served.",
    "Thank you for being a loyal customer.",
    "Failure to respond will result in account closure.",
    "This is not a scam, fully verified and legit.",
]
links = ["bit.ly/claim-now", "secure-verify-account.com", "win-big-prizes.net",
         "paypa1-support.com", "free-gift-cards.biz", "update-billing-now.info"]
emails = ["claims@lottery-intl.com", "support@account-verify.net", "prize@bigwin.biz"]
phones = ["1-800-555-0199", "1-888-222-9090", "1-877-444-6161"]

def make_spam():
    o = random.choice(spam_openers)
    hook = random.choice(spam_hooks).format(
        amount=random.choice([500, 1000, 2500, 10000, 50, 20, 99]),
        prize=random.choice(spam_prizes),
    )
    cta = random.choice(spam_ctas).format(
        link=random.choice(links), phone=random.choice(phones), email=random.choice(emails)
    )
    closer = random.choice(spam_closers)
    extra = random.choice([
        "", " Limited time only.", " Offer valid while supplies last.",
        " No purchase necessary.", " Winners will be notified by email.",
    ])
    subject_variants = [
        f"{o}: {hook}",
        f"Re: {hook}",
        f"{o}!!! {hook}",
    ]
    body = f"{random.choice(subject_variants)}. {cta}. {closer}{extra}"
    return body

# ---------- HAM (legitimate) building blocks ----------
ham_senders = ["mom", "dad", "my manager", "the HR team", "our professor",
               "the project lead", "my landlord", "the IT helpdesk", "a coworker",
               "my sister", "the finance team", "our client"]
ham_topics = [
    "the quarterly report is attached, let me know if you have questions",
    "can we reschedule our meeting to Thursday at 3pm?",
    "thanks for sending over the notes from yesterday's call",
    "here is the invoice for last month's services",
    "just checking in to see how the project is going",
    "the flight details for our trip are confirmed, see attached itinerary",
    "reminder that rent is due on the 1st of the month",
    "the server maintenance window is scheduled for this weekend",
    "great job on the presentation today, the client loved it",
    "let's grab lunch next week to catch up",
    "the code review comments are ready, please take a look when you can",
    "happy birthday! hope you have a wonderful day",
    "attached is the updated syllabus for next semester",
    "your order has shipped and will arrive in 3-5 business days",
    "the team standup is moved to 10am starting tomorrow",
    "I reviewed the document and left a few comments in the margins",
    "can you send me the file you mentioned during the call?",
    "the dentist appointment is confirmed for next Tuesday at 9am",
    "we finished the migration ahead of schedule, no downtime reported",
    "here's a recap of what we discussed in the retro",
]
ham_closers = [
    "Thanks!", "Best regards,", "Talk soon.", "Let me know if you need anything else.",
    "Looking forward to hearing from you.", "Cheers,", "Have a great day!",
    "Appreciate your help with this.", "See you then.", "",
]
ham_signoffs = ["Sarah", "James", "the Support Team", "Alex", "Priya", "Mike", "Dana"]

def make_ham():
    sender = random.choice(ham_senders)
    topic = random.choice(ham_topics)
    closer = random.choice(ham_closers)
    signoff = random.choice(ham_signoffs)
    templates = [
        f"Hi, this is a note from {sender} - {topic}. {closer}",
        f"Hey, {topic}. {closer} - {signoff}",
        f"Hello, {topic}. {closer}",
        f"{topic.capitalize()}. {closer} - {signoff}",
    ]
    return random.choice(templates)

N = 900
rows = []
for _ in range(N):
    rows.append((make_spam(), "spam"))
for _ in range(N):
    rows.append((make_ham(), "ham"))

random.shuffle(rows)

with open("/home/claude/spam-detector/project/dataset.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["text", "label"])
    w.writerows(rows)

print(f"Wrote {len(rows)} rows to dataset.csv")
