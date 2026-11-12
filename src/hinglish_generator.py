"""Generate the SYNTHETIC Hinglish SMS supplement.

IMPORTANT - these messages are synthetic. They were written by the project team as
templates that imitate publicly reported Indian SMS scam patterns (fake KYC, UPI fraud,
account blocking, lottery, fake jobs, OTP requests, fake delivery) and everyday
legitimate Hinglish messages. They are NOT real collected SMS and must not be reported
as real-world data.

Privacy safeguards:
* No real names: only generic forms of address (bhai, sir, beta, customer ...).
* Phone numbers use the +91 55555 xxxxx range (5xxxx numbers are not allocated to
  Indian mobile subscribers) and are masked anyway during preprocessing.
* URLs use the reserved ``.example`` top-level domain (RFC 2606), so no link is real.
* OTPs/amounts are random and do not belong to anyone.

Every message keeps its ``template_id``. Splits are made by template so that variants of
the same template never appear in both the training and the test set (prevents leakage).
"""
from __future__ import annotations

import random
import re

import pandas as pd

from src.config import HINGLISH_DIR, SEED

BANKS = ["SBI", "HDFC", "ICICI", "PNB", "Axis Bank", "Bank of Baroda", "Kotak", "Canara Bank"]
WALLETS = ["Paytm", "PhonePe", "GPay", "BHIM UPI", "UPI"]
COURIERS = ["India Post", "Delhivery", "BlueDart", "DTDC", "Ekart"]
SHOPS = ["Flipkart", "Amazon", "Myntra", "Meesho", "Ajio", "Big Bazaar", "DMart"]
TELCOS = ["Jio", "Airtel", "Vi", "BSNL"]
CITIES = ["Pune", "Mumbai", "Baramati", "Nagpur", "Nashik", "Satara", "Kolhapur"]
ADDRESS = ["bhai", "yaar", "sir", "beta", "dost", "didi", "bro"]
DAYS = ["kal", "aaj", "parso", "Monday ko", "Friday ko", "Sunday ko"]
TIMES = ["10 baje", "11 baje", "2 baje", "4 baje", "6 baje", "7:30 baje", "9 baje"]
SUBJECTS = ["DBMS", "NLP", "Data Mining", "Maths", "Physics", "Machine Learning", "Networks"]
FOODS = ["chai", "samosa", "pizza", "biryani", "vada pav", "misal"]
PHISH_WORDS = ["verify", "update", "kyc", "secure", "login", "reactivate", "refund", "claim", "reward"]


def _url(rng: random.Random) -> str:
    word = rng.choice(PHISH_WORDS)
    tag = "".join(rng.choice("abcdefghjkmnpqrstuvwxyz23456789") for _ in range(5))
    return rng.choice([
        f"http://{word}-{tag}.example/{rng.choice(['login', 'kyc', 'verify', 'pay'])}",
        f"https://{tag}.example/{word}",
        f"bit-{tag}.example",
        f"www.{word}{tag}.example",
    ])


def _legit_url(rng: random.Random) -> str:
    return rng.choice(["www.college-portal.example", "https://results.example/semester",
                       "www.irctc-ticket.example/pnr", "https://library.example/renew"])


def _phone(rng: random.Random) -> str:
    return "+91 55555 " + "".join(str(rng.randint(0, 9)) for _ in range(5))


SLOT_FILLERS = {
    "bank": lambda r: r.choice(BANKS), "wallet": lambda r: r.choice(WALLETS),
    "courier": lambda r: r.choice(COURIERS), "shop": lambda r: r.choice(SHOPS),
    "telco": lambda r: r.choice(TELCOS), "city": lambda r: r.choice(CITIES),
    "addr": lambda r: r.choice(ADDRESS), "day": lambda r: r.choice(DAYS),
    "time": lambda r: r.choice(TIMES), "subject": lambda r: r.choice(SUBJECTS),
    "food": lambda r: r.choice(FOODS), "url": _url, "legit_url": _legit_url, "phone": _phone,
    "amount": lambda r: f"Rs {r.choice([499, 999, 1500, 2500, 4999, 9999, 25000, 50000])}",
    "big_amount": lambda r: f"Rs {r.choice(['5,00,000', '10 lakh', '25 lakh', '1 crore', '50,000'])}",
    "otp": lambda r: str(r.randint(100000, 999999)),
    "hours": lambda r: str(r.choice([2, 6, 12, 24])),
    "pct": lambda r: str(r.choice([10, 20, 30, 40, 50, 60, 70])),
    "salary": lambda r: f"Rs {r.choice([15000, 25000, 30000, 45000])}",
    "order": lambda r: "OD" + "".join(str(r.randint(0, 9)) for _ in range(6)),
}

# (template_id, label, category, template text)
TEMPLATES: list[tuple[str, str, str, str]] = [
    # ------------------------------------------------------------------ PHISHING
    ("P01", "phishing", "kyc", "Aapka {bank} account band ho jayega. KYC update karne ke liye abhi link open karo: {url}"),
    ("P02", "phishing", "kyc", "Dear customer, aapka {bank} KYC expire ho gaya hai. {hours} ghante me update nahi kiya to account block hoga. Click {url}"),
    ("P03", "phishing", "kyc", "{bank} ALERT: PAN card link nahi hai, aapki netbanking aaj raat band ho jayegi. Turant update karein {url}"),
    ("P04", "phishing", "kyc", "Aapka {telco} SIM {hours} ghante me block ho jayega kyunki KYC pending hai. Verify karne ke liye call karo {phone}"),
    ("P05", "phishing", "kyc", "Dear {bank} user, aapka Aadhaar verification fail hua hai. Account chalu rakhne ke liye details yaha bharo {url}"),
    ("P06", "phishing", "account_block", "Aapka {bank} debit card block kar diya gaya hai. Unblock karne ke liye card number aur PIN reply karo"),
    ("P07", "phishing", "account_block", "URGENT: aapke {bank} account me suspicious login hua hai. Abhi password reset karo warna paisa kat jayega {url}"),
    ("P08", "phishing", "account_block", "{bank}: aapka account temporarily suspend hai. Reactivate karne ke liye login karein {url} jaldi"),
    ("P09", "phishing", "account_block", "Aapka electricity connection aaj raat 9:30 baje kat jayega kyunki pichla bill update nahi hua. Turant call karo {phone}"),
    ("P10", "phishing", "upi", "{wallet}: aapko {amount} cashback mila hai. Paisa lene ke liye UPI PIN daal kar request accept karo {url}"),
    ("P11", "phishing", "upi", "Aapke {wallet} me {amount} ka refund pending hai. Claim karne ke liye apna UPI PIN aur OTP share karein"),
    ("P12", "phishing", "upi", "{addr} galti se maine aapke number par {amount} bhej diye, please wapas kar do, maine collect request bheja hai usme PIN daal do"),
    ("P13", "phishing", "upi", "Congratulations! {wallet} reward {amount} aapke account me credit hone wala hai. Abhi {url} pe jaake verify karo"),
    ("P14", "phishing", "otp", "{bank} customer care: aapke account ki security ke liye abhi aaya hua OTP is number pe bhejo {phone}"),
    ("P15", "phishing", "otp", "Aapka parcel deliver karne ke liye ek OTP aaya hai, delivery boy ko batana zaroori hai. Abhi wo OTP reply karo"),
    ("P16", "phishing", "otp", "Hello {addr}, mera phone kharab hai, aapke number par mera OTP gaya hai, please jaldi forward kar do"),
    ("P17", "phishing", "otp", "{bank} reward points expire ho rahe hai. Redeem karne ke liye OTP aur card ki last 4 digit reply karein"),
    ("P18", "phishing", "delivery", "{courier}: aapka parcel address incomplete hone ki wajah se ruk gaya hai. Rs 25 fee pay karke address update karo {url}"),
    ("P19", "phishing", "delivery", "Aapka {shop} order {order} return ho jayega. Delivery confirm karne ke liye card details yaha daalo {url}"),
    ("P20", "phishing", "delivery", "{courier} alert: customs charge pending hai, parcel aaj destroy ho jayega. Abhi payment karo {url}"),
    ("P21", "phishing", "lottery", "Badhai ho! Aapne {telco} lucky draw me {big_amount} jeete hai. Paisa lene ke liye processing fee {amount} bhejo aur bank details do"),
    ("P22", "phishing", "lottery", "KBC lottery winner! Aapka number select hua hai {big_amount} ke liye. Claim karne ke liye call karo {phone} aur account number batao"),
    ("P23", "phishing", "lottery", "Aapka mobile number {big_amount} ke international lottery me jeeta hai. Apna naam, Aadhaar aur bank details reply karein"),
    ("P24", "phishing", "job", "Part time job! Ghar baithe roz {amount} kamao. Registration ke liye {amount} fee bharo aur bank details bhejo {url}"),
    ("P25", "phishing", "job", "Aapka resume select hua hai {shop} me {salary} salary ke liye. Joining confirm karne ke liye Rs 2000 security deposit bhejo {url}"),
    ("P26", "phishing", "job", "Work from home: sirf YouTube videos like karo aur {amount} roz pao. Telegram join karo aur UPI ID verify karo {url}"),
    ("P27", "phishing", "tax_refund", "Income Tax Dept: aapka {amount} ka refund approve hua hai. Account verify karke refund lo {url}"),
    ("P28", "phishing", "tax_refund", "Aapka gas subsidy {amount} pending hai. Bank details update karne ke liye link kholo {url}"),
    ("P29", "phishing", "loan", "Pre-approved loan {big_amount} sirf 5 min me! Abhi apply karo, PAN aur net banking password se verify karo {url}"),
    ("P30", "phishing", "loan", "Aapka loan approve ho gaya hai. Amount release karne ke liye processing charge {amount} UPI karo aur OTP batao"),
    ("P31", "phishing", "impersonation", "Papa main naye number se message kar raha hu, urgent {amount} chahiye, is UPI pe bhej do jaldi, kisi ko mat batana"),
    ("P32", "phishing", "impersonation", "Police cyber cell: aapke Aadhaar se illegal parcel bheja gaya hai. Case band karne ke liye abhi call karo {phone}"),
    # ------------------------------------------------------------------ SPAM (promotional / bulk, no credential theft)
    ("S01", "spam", "promo", "Mega SALE! {shop} par {pct}% tak OFF sirf aaj. Jaldi shopping karo, offer limited hai!"),
    ("S02", "spam", "promo", "Diwali dhamaka offer: har order par flat {pct}% discount. {shop} app download karo aur khareedo"),
    ("S03", "spam", "promo", "Weekend special: {food} combo sirf Rs 99 me. Aaj hi order karo, free delivery!"),
    ("S04", "spam", "promo", "Naye saal ki khushi me {shop} lekar aaya hai buy 1 get 1 free. Stock khatam hone se pehle aao!"),
    ("S05", "spam", "telecom", "{telco}: sirf Rs 299 me unlimited calls aur 2GB/day data. Abhi recharge karo aur maza lo"),
    ("S06", "spam", "telecom", "Dear customer, {telco} ka naya plan aaya hai, 84 din validity ke saath. Recharge karne ke liye *121# dial karo"),
    ("S07", "spam", "telecom", "{telco} caller tune set karo sirf Rs 5 me! Apna favourite gaana chuno, reply SET to start"),
    ("S08", "spam", "realestate", "{city} me 2BHK flat sirf 35 lakh me! Site visit free, aaj hi call karo {phone}"),
    ("S09", "spam", "realestate", "Plot booking shuru! {city} highway ke paas NA plot, easy EMI available. Visit karo is Sunday"),
    ("S10", "spam", "education", "IIT JEE/NEET crash course admission open! {city} me best faculty, {pct}% scholarship. Abhi register karo"),
    ("S11", "spam", "education", "English speaking course sirf 30 din me! Free demo class kal 5 baje, seat book karo {phone}"),
    ("S12", "spam", "health", "Weight kam karo sirf 15 din me ayurvedic powder se! Order karo aaj, COD available"),
    ("S13", "spam", "health", "Free health checkup camp {day} {city} me. Sugar, BP check bilkul free, sab log aaiye"),
    ("S14", "spam", "finance", "Credit card lo zero joining fee ke saath! Lifetime free card, apply karne ke liye reply YES"),
    ("S15", "spam", "finance", "Mutual fund me SIP shuru karo sirf Rs 500 se. Expert advice ke liye call karo {phone}"),
    ("S16", "spam", "finance", "Gold loan sabse kam byaj par! Nearest branch visit karo, same day disbursal"),
    ("S17", "spam", "entertainment", "Naya web series dekho free! Sirf aaj ke liye OTT subscription {pct}% off, abhi subscribe karo"),
    ("S18", "spam", "entertainment", "Cricket fantasy league join karo aur bade inaam jeeto! Pehla contest free, app download karo"),
    ("S19", "spam", "entertainment", "Rashifal sunne ke liye abhi call karo, pehle 3 minute free! Apna bhavishya jaano"),
    ("S20", "spam", "promo", "Flash sale alert: smartphones par {pct}% off, sirf agle {hours} ghante! {shop} par check karo"),
    ("S21", "spam", "promo", "Shaadi season offer: lehenga aur sherwani par {pct}% tak chhoot. {city} showroom visit karein"),
    ("S22", "spam", "services", "AC service sirf Rs 399 me! Garmi se pehle booking karo, technician ghar aayega"),
    ("S23", "spam", "services", "Ghar baithe car wash service, monthly plan par {pct}% discount. Book karne ke liye reply WASH"),
    ("S24", "spam", "political", "Aapke ward ke vikas ke liye {day} sabha hai, sab log zaroor aaye. Free nashta aur chai"),
    ("S25", "spam", "promo", "Restaurant opening offer! {city} me naya outlet, pehle 100 customers ko free {food}"),
    ("S26", "spam", "telecom", "Data khatam? {telco} 1GB booster sirf Rs 19 me. Abhi activate karo app se"),
    # ------------------------------------------------------------------ HAM (legitimate)
    ("H01", "ham", "college", "Kal college lecture {time} hai, time pe aa jana."),
    ("H02", "ham", "college", "{addr}, {subject} ka assignment kab tak submit karna hai? Mujhe date yaad nahi"),
    ("H03", "ham", "college", "Aaj {subject} ki practical cancel ho gayi hai, sir ne bola kal hogi"),
    ("H04", "ham", "college", "Exam form bharne ki last date {day} hai, sab log jaldi bhar do"),
    ("H05", "ham", "college", "{addr} notes bhej de na {subject} ke, kal padhai karni hai"),
    ("H06", "ham", "college", "Project meeting {day} {time} library me, laptop leke aana"),
    ("H07", "ham", "family", "Beta khana kha liya? Ghar kab aa rahe ho?"),
    ("H08", "ham", "family", "Mummy bol rahi hai {day} sab log mama ke ghar ja rahe hai, tum bhi chalo"),
    ("H09", "ham", "family", "Papa ki train {time} pahuchegi, station pe lene chale jaana"),
    ("H10", "ham", "family", "Doodh aur bread le aana aate waqt, ghar me khatam ho gaya hai"),
    ("H11", "ham", "friends", "{addr} aaj shaam {food} khane chalein? {time} milte hai"),
    ("H12", "ham", "friends", "Match dekhne aa raha hai kya? {day} {time} start hai"),
    ("H13", "ham", "friends", "Happy birthday {addr}! Party kab de raha hai?"),
    ("H14", "ham", "friends", "Main thoda late ho jaunga, traffic bahut hai, 15 min me pahuchta hu"),
    ("H15", "ham", "friends", "Movie ki tickets book kar di hai, {day} {time} ka show hai"),
    ("H16", "ham", "work", "Sir, aaj ki meeting {time} shift ho gayi hai, please note kar lijiye"),
    ("H17", "ham", "work", "Report bhej diya hai mail pe, ek baar check kar lena"),
    ("H18", "ham", "work", "Kal office {time} tak pahuch jaana, client visit hai"),
    ("H19", "ham", "transaction", "Aapke {bank} account se {amount} debit hue hai UPI transaction me. Agar aapne nahi kiya to bank branch se sampark karein"),
    ("H20", "ham", "transaction", "Aapka OTP {otp} hai. Kisi ke saath share na karein. {bank} kabhi OTP nahi maangta"),
    ("H21", "ham", "transaction", "{wallet}: aapne {amount} successfully bheje hai. Dhanyavaad!"),
    ("H22", "ham", "transaction", "Aapka {shop} order {order} deliver ho gaya hai. Shopping ke liye dhanyavaad"),
    ("H23", "ham", "transaction", "{courier}: aapka parcel aaj deliver hoga. Delivery ke time ghar par rahiye"),
    ("H24", "ham", "transaction", "Bijli bill {amount} successfully pay ho gaya hai. Receipt app me dekh sakte hai"),
    ("H25", "ham", "appointment", "Aapka doctor appointment {day} {time} confirm hai. Time pe aaiye"),
    ("H26", "ham", "appointment", "Aapki train ka PNR confirm ho gaya hai, details {legit_url} pe dekh sakte hai"),
    ("H27", "ham", "appointment", "Library book return karne ki last date {day} hai, renew ke liye {legit_url}"),
    ("H28", "ham", "college", "Semester result declare ho gaya hai, {legit_url} pe check karo"),
    ("H29", "ham", "friends", "Haan {addr} theek hai, kal baat karte hai, abhi busy hu"),
    ("H30", "ham", "family", "Didi ki shaadi ki date fix ho gayi hai, sab ko bata dena"),
    ("H31", "ham", "work", "Aaj work from home kar raha hu, kuch urgent ho to call karna"),
    ("H32", "ham", "friends", "Gym chalein kal subah? {time} wala batch theek rahega"),
]

STYLE_SWAPS = [("aap", "ap"), ("nahi", "nhi"), ("karo", "kro"), ("hai", "h"), ("please", "pls"),
               ("jaldi", "jldi"), ("abhi", "abi")]


def _fill(template: str, rng: random.Random) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: SLOT_FILLERS[m.group(1)](rng), template)


def _stylise(text: str, rng: random.Random) -> str:
    """Add realistic SMS variation: chat spellings, casing, punctuation."""
    for full, short in STYLE_SWAPS:
        if rng.random() < 0.3:
            text = re.sub(rf"\b{full}\b", short, text)
    r = rng.random()
    if r < 0.12:
        text = text.upper()
    elif r < 0.30:
        text = text.lower()
    if rng.random() < 0.2:
        text = text.rstrip(".!") + rng.choice(["!!", "...", " !", ""])
    return text


def generate(variants_per_template: int = 5, seed: int = SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    rows, seen = [], set()
    for tid, label, category, template in TEMPLATES:
        made, attempts = 0, 0
        while made < variants_per_template and attempts < 50:
            attempts += 1
            text = _stylise(_fill(template, rng), rng)
            if text.lower() in seen:
                continue
            seen.add(text.lower())
            rows.append({"text": text, "label": label, "category": category, "template_id": tid,
                         "source": "synthetic_hinglish", "language": "hinglish"})
            made += 1
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    out = HINGLISH_DIR / "hinglish_sms_synthetic.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} synthetic Hinglish messages to {out}")
    print(df.label.value_counts().to_string())
    print(df.groupby("label").template_id.nunique().to_string())
