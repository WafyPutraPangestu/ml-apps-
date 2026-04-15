import pandas as pd
import random
from datetime import datetime, timedelta
import os

TEMPLATES = {
    'Tinggi': {
        'judul': [
            'Internet mati total sejak tadi malam',
            'Server down, semua user tidak bisa akses',
            'Sistem error darurat, operasional terhenti',
            'Koneksi terputus total lebih dari 3 jam',
            'Database tidak bisa diakses sama sekali',
            'Gangguan jaringan parah, kantor tidak bisa bekerja',
            'Email server down urgent',
            'VPN tidak bisa connect, WFH terganggu',
        ],
        'deskripsi': [
            'Sudah {n} jam tidak bisa akses internet sama sekali. Seluruh tim tidak bisa bekerja.',
            'Server down mendadak sejak pukul 08.00. Semua user terdampak, aktivitas bisnis berhenti.',
            'Koneksi putus total. Sudah coba restart router tapi tidak membantu. Sangat urgent.',
            'Error 503 terus muncul. Database tidak bisa diakses. Sudah {n} jam.',
            'Sistem kami lumpuh total. Tolong segera ditangani karena berdampak ke pelanggan kami.',
        ],
        'kategori_gangguan': ['Gangguan Jaringan', 'Gangguan Sistem', 'Gangguan Jaringan', 'Gangguan Bandwidth'],
        'kategori_pelanggan_weights': {'Perusahaan': 0.65, 'UMKM': 0.25, 'Rumah': 0.10},
    },
    'Sedang': {
        'judul': [
            'Internet sangat lambat sejak kemarin',
            'Koneksi tidak stabil, sering putus',
            'Loading aplikasi sangat lama',
            'Upload file selalu gagal',
            'Video call sering lag dan putus',
            'Kecepatan internet jauh di bawah paket',
            'Streaming buffering terus',
            'Koneksi fluktuatif sepanjang hari',
        ],
        'deskripsi': [
            'Kecepatan internet sangat lambat, tidak sesuai paket yang dibayar. Sudah {n} hari.',
            'Koneksi sering terputus setiap {n} menit. Sangat mengganggu pekerjaan.',
            'Loading halaman web dan aplikasi sangat lama, padahal dulu normal.',
            'Upload file berukuran besar selalu gagal di tengah jalan.',
            'Video call dengan klien sering putus dan laggy, sangat memalukan.',
            'Speedtest menunjukkan hanya {n} Mbps padahal paket 100 Mbps.',
        ],
        'kategori_gangguan': ['Gangguan Bandwidth', 'Gangguan Jaringan', 'Gangguan Bandwidth', 'Gangguan Sistem'],
        'kategori_pelanggan_weights': {'Perusahaan': 0.40, 'UMKM': 0.35, 'Rumah': 0.25},
    },
    'Rendah': {
        'judul': [
            'Pertanyaan cara setting router',
            'Minta info upgrade paket internet',
            'Cara ganti password WiFi',
            'Request penambahan akses user',
            'Konsultasi paket untuk kantor baru',
            'Tanya jadwal maintenance rutin',
            'Info biaya tambah bandwidth',
            'Cara setting email di Outlook',
        ],
        'deskripsi': [
            'Mau tanya bagaimana cara setting router agar optimal.',
            'Ingin konsultasi untuk upgrade ke paket yang lebih tinggi.',
            'Bisa bantu panduan cara ganti password WiFi secara mandiri?',
            'Mohon dibuatkan akses untuk {n} user baru yang baru bergabung.',
            'Kami akan buka kantor baru, mau konsultasi paket yang cocok.',
            'Minta info jadwal maintenance agar bisa siapkan backup.',
        ],
        'kategori_gangguan': ['Pertanyaan Teknis', 'Request Layanan', 'Pertanyaan Teknis'],
        'kategori_pelanggan_weights': {'Rumah': 0.45, 'UMKM': 0.35, 'Perusahaan': 0.20},
    }
}

def weighted_choice(weight_dict):
    keys = list(weight_dict.keys())
    weights = list(weight_dict.values())
    return random.choices(keys, weights=weights)[0]

def generate_dataset(n=1000):
    data = []
    priorities = ['Tinggi'] * int(n*0.25) + ['Sedang'] * int(n*0.45) + ['Rendah'] * int(n*0.30)
    random.shuffle(priorities)

    print(f"Generating {n} records...")

    for i, prioritas in enumerate(priorities):
        if (i+1) % 200 == 0:
            print(f"  Progress: {i+1}/{n}")

        t = TEMPLATES[prioritas]
        judul = random.choice(t['judul'])
        deskripsi = random.choice(t['deskripsi']).format(n=random.randint(1, 14))
        kategori = random.choice(t['kategori_gangguan'])
        pelanggan = weighted_choice(t['kategori_pelanggan_weights'])

        # Noise 8%: simulasi human error labeling
        if random.random() < 0.08:
            noise_map = {'Tinggi': 'Sedang', 'Sedang': random.choice(['Tinggi', 'Rendah']), 'Rendah': 'Sedang'}
            prioritas = noise_map[prioritas]

        days_ago = random.randint(0, 90)
        waktu = datetime.now() - timedelta(days=days_ago, hours=random.randint(0, 23))

        data.append({
            'judul': judul,
            'deskripsi': deskripsi,
            'kategori_gangguan': kategori,
            'kategori_pelanggan': pelanggan,
            'waktu_lapor': waktu.strftime('%Y-%m-%d %H:%M:%S'),
            'prioritas': prioritas
        })

    return pd.DataFrame(data).sample(frac=1).reset_index(drop=True)

if __name__ == "__main__":
    print("=" * 50)
    print("GENERATING DUMMY DATASET")
    print("=" * 50)

    df = generate_dataset(1000)

    # Simpan ke folder yang sama dengan script ini
    output_path = os.path.join(os.path.dirname(__file__), "training_data.csv")
    df.to_csv(output_path, index=False)

    print()
    print("=" * 50)
    print("✅ DATASET GENERATED!")
    print("=" * 50)
    print(f"Total     : {len(df)} rows")
    print(f"File      : {output_path}")
    print("\nDistribusi:")
    print(df['prioritas'].value_counts())
    print("\nSample:")
    print(df.head(3))