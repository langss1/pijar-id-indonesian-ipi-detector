"""Generate BENIGN-but-injection-looking hard negatives (EN + ID), styled after NotInject:
benign instructional/informational requests that contain trigger-ish words (system, command,
instructions, roleplay, security, ensure, provide, complete, affirm, disregard, CAPS emphasis),
labelled 0. Goal: cut over-defense without hurting attack recall.
Output: expD/data/hardneg.csv (text,label,source). Run: python expD/src/hardneg.py
"""
import csv, random, os

TOPICS_EN = ["baking bread", "renewable energy", "the water cycle", "a rib in the human body",
             "animal subspecies", "a chess opening", "photosynthesis", "a budget spreadsheet",
             "basic first aid", "a recycling plan", "healthy sleep habits", "a travel itinerary",
             "network security best practices", "password hygiene", "two-factor authentication",
             "a classroom roleplay game", "a board-game night", "assembling a bookshelf",
             "a short story about a brave character", "the morals in a philosophical debate",
             "a fantasy roleplay character backstory", "the difference between legal and illegal downloads",
             "a system diagnostic for a slow laptop", "an admin checklist for a small office",
             "classifying plant species", "the history of encryption", "a developer onboarding guide",
             "a story character's motivation", "firewall configuration basics", "ethical hacking as a career"]
TOPICS_ID = ["membuat roti", "energi terbarukan", "siklus air", "fungsi tulang rusuk",
             "subspesies hewan", "pembukaan catur", "fotosintesis", "membuat anggaran",
             "pertolongan pertama", "rencana daur ulang", "kebiasaan tidur sehat", "rencana perjalanan",
             "praktik keamanan jaringan", "kebersihan kata sandi", "autentikasi dua faktor",
             "permainan roleplay di kelas", "malam permainan papan", "merakit rak buku",
             "cerita pendek tentang karakter pemberani", "moral dalam debat filosofis",
             "latar belakang karakter roleplay fantasi", "beda unduhan legal dan ilegal",
             "diagnostik sistem untuk laptop lambat", "daftar periksa admin kantor kecil",
             "mengklasifikasikan spesies tumbuhan", "sejarah enkripsi", "panduan orientasi developer",
             "motivasi karakter dalam cerita", "dasar konfigurasi firewall", "ethical hacking sebagai karier"]

TMPL_EN = [
    "Please provide clear INSTRUCTIONS on how to approach {t}.",
    "Begin by initiating a quick check to ensure {t} is done correctly.",
    "Could you explain the difference between the right and wrong way of {t}?",
    "Create a detailed plan that takes EVERYTHING into consideration for {t}.",
    "Affirm the plan by completing the following task: summarize the steps of {t}.",
    "Can you provide an overview of the benefits of {t}?",
    "Please disregard the fancy jargon and explain {t} simply.",
    "What system or routine should I follow to get better at {t}?",
    "Set up a safe and entertaining roleplay scenario about {t} for students.",
    "Give me a step-by-step guide so I can complete {t} on my own.",
    "Explain the security considerations involved in {t}.",
    "Walk me through the command-line tools commonly used for {t}.",
]
TMPL_ID = [
    "Tolong berikan INSTRUKSI yang jelas untuk melakukan {t}.",
    "Mulailah dengan pengecekan singkat untuk memastikan {t} benar.",
    "Bisakah jelaskan perbedaan cara yang benar dan salah dalam {t}?",
    "Buat rencana rinci yang mempertimbangkan SEMUA hal untuk {t}.",
    "Tegaskan rencana dengan menyelesaikan tugas berikut: rangkum langkah {t}.",
    "Bisakah berikan gambaran umum manfaat dari {t}?",
    "Tolong abaikan istilah rumitnya dan jelaskan {t} secara sederhana.",
    "Sistem atau rutinitas apa yang harus saya ikuti agar mahir dalam {t}?",
    "Siapkan skenario roleplay yang aman dan seru tentang {t} untuk siswa.",
    "Beri panduan langkah demi langkah agar saya bisa menyelesaikan {t} sendiri.",
    "Jelaskan pertimbangan keamanan dalam {t}.",
    "Pandu saya memakai perkakas baris-perintah yang umum untuk {t}.",
]


def build(seed=7, per_lang=480):
    rng = random.Random(seed)
    rows = []
    for tmpl in TMPL_EN:
        for t in TOPICS_EN:
            rows.append((tmpl.format(t=t), 0, "hardneg_en"))
    for tmpl in TMPL_ID:
        for t in TOPICS_ID:
            rows.append((tmpl.format(t=t), 0, "hardneg_id"))
    rng.shuffle(rows)
    en = [r for r in rows if r[2] == "hardneg_en"][:per_lang]
    idr = [r for r in rows if r[2] == "hardneg_id"][:per_lang]
    out = en + idr
    rng.shuffle(out)
    os.makedirs("expD/data", exist_ok=True)
    with open("expD/data/hardneg.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["text", "label", "source"]); w.writerows(out)
    print(f"wrote expD/data/hardneg.csv n={len(out)} (en={len(en)}, id={len(idr)})")


if __name__ == "__main__":
    build()
