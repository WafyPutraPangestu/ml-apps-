import pandas as pd
import random
from datetime import datetime, timedelta

print("="*50)
print("GENERATING DUMMY DATASET")
print("="*50)
print()

TEMPLATES = {
    'Tinggi': {
        'judul': [
            'Internet mati total',
            'Server down urgent',
            'Sistem error darurat',
            'Koneksi terputus',
            'Database tidak bisa diakses'
        ],
        'deskripsi': [
            'Sudah {n} jam tidak bisa akses',
            'Sistem error sejak pagi',
            'Koneksi putus terus',
            'Server down semua user tidak bisa akses',
            'Error terus muncul'
        ]
    },
    'Sedang': {
        'judul': [
            'Internet lambat',
            'Koneksi tidak stabil',
            'Loading lama',
            'Sering terputus',
            'Upload gagal'
        ],
        'deskripsi': [
            'Internet agak lambat',
            'Koneksi sering terputus',
            'Loading aplikasi lama',
            'Koneksi tidak stabil',
            'Upload file gagal'
        ]
    },
    'Rendah': {
        'judul': [
            'Pertanyaan setting',
            'Cara ganti password',
            'Info layanan',
            'Request akses',
            'Konsultasi'
        ],
        'deskripsi': [
            'Mau tanya cara setting',
            'Bisa bantu ganti password?',
            'Mohon info upgrade paket',
            'Mau konsultasi fitur',
            'Request akses user baru'
        ]
    }
}

KATEGORI = ['Gangguan Jaringan', 'Gangguan Bandwidth', 'Gangguan Sistem', 
            'Pertanyaan Teknis', 'Request Layanan']
PELANGGAN = ['Rumah', 'Perusahaan', 'UMKM']

def generate_dataset(n=1000):
    data = []
    priorities = ['Tinggi']*300 + ['Sedang']*400 + ['Rendah']*300
    random.shuffle(priorities)
    
    print(f"Generating {n} records...")
    
    for i, prioritas in enumerate(priorities):
        if (i+1) % 200 == 0:
            print(f"Progress: {i+1}/{n}")
        
        judul = random.choice(TEMPLATES[prioritas]['judul'])
        desc = random.choice(TEMPLATES[prioritas]['deskripsi'])
        deskripsi = desc.format(n=random.randint(2, 12))
        
        if prioritas == 'Tinggi':
            kategori = random.choice(['Gangguan Jaringan', 'Gangguan Sistem'])
            pelanggan = random.choices(PELANGGAN, weights=[0.2, 0.6, 0.2])[0]
        else:
            kategori = random.choice(KATEGORI)
            pelanggan = random.choice(PELANGGAN)
        
        days_ago = random.randint(0, 60)
        waktu = datetime.now() - timedelta(days=days_ago)
        
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
    df = generate_dataset(1000)
    df.to_csv("training_data.csv", index=False)
    
    print()
    print("="*50)
    print("✅ DATASET GENERATED!")
    print("="*50)
    print(f"\nTotal: {len(df)} rows")
    print("\nDistribution:")
    print(df['prioritas'].value_counts())
    print("\nSample:")
    print(df.head())
    print(f"\nFile: training_data.csv")