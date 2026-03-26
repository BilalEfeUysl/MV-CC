import pandas as pd
import matplotlib.pyplot as plt
import os

# Dosya adımız
csv_dosyasi = 'training_progress.csv'

def grafik_olustur():
    # Dosyanın var olup olmadığını kontrol edelim
    if not os.path.exists(csv_dosyasi):
        print(f"Hata: '{csv_dosyasi}' bulunamadı! Lütfen Python dosyası ile aynı klasörde olduğundan emin ol.")
        return

    print(f"'{csv_dosyasi}' okunuyor...")
    # CSV dosyasını okuyoruz
    df = pd.read_csv(csv_dosyasi)
    
    # Güvenlik: Sütun isimlerinde kazara oluşmuş boşluklar varsa temizleyelim
    df.columns = df.columns.str.strip()
    
    # --- LOSS (KAYIP) GRAFİĞİ ---
    plt.figure(figsize=(10, 6))
    
    # X ekseni olarak 'Step' kullanıyoruz çünkü henüz 0. Epoch içindeyiz
    x_ekseni = df['Step']
    
    # Çizgilerimizi ekliyoruz (Büyük harflere dikkat!)
    plt.plot(x_ekseni, df['Train_Loss'], label='Eğitim Kaybı (Train Loss)', color='blue', marker='o', linewidth=2)
    plt.plot(x_ekseni, df['Val_Loss'], label='Doğrulama Kaybı (Val Loss)', color='red', marker='s', linewidth=2)

    # Grafiği şekillendiriyoruz
    plt.title('Model Öğrenme Eğrisi (Loss Curve)', fontsize=14, fontweight='bold')
    plt.xlabel('Step (Eğitim Adımı)', fontsize=12)
    plt.ylabel('Hata Payı (Loss)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Grafiği aynı klasöre yüksek çözünürlüklü bir resim olarak kaydet
    kayit_adi = 'loss_grafigi_local.png'
    plt.savefig(kayit_adi, dpi=300, bbox_inches='tight')
    print(f"✅ Loss grafiği başarıyla oluşturuldu ve '{kayit_adi}' olarak bu klasöre kaydedildi.")
    
    # Grafiği ekranda pencere olarak açıp göster
    plt.show()

if __name__ == "__main__":
    grafik_olustur()