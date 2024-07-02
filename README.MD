# FAITH Takımının Online Yarışma Sürecinde Kullanacağı GitHub Reposu

Bundan sonraki sürece iş bölümü şeklinde devam edeceğiz. </br>
Ali Kemal - Arif Tuna   --> Object Detection </br>
Eren      - Mehmet Akif --> Lokasyon 


3 Farklı Branch olacak:
- main
- location
- objdet

Dosyalar üzerinde kendi branch'ınızda değişiklik yapıp commit atarsınız. </br>
Son halini aldığında yada çakışma olmadığında main branch'le marge edip devam edilir.





# Kurulum
- Sistem için bir sanal ortam oluşturulur. (Anaconda ile.)
```shell
$ conda create -n faith python=3.7
$ conda activate faith
```
- Gerekli paketler yüklenir.
```shell
$ pip install -r requirements.txt
```

- Sistemi çalıştırmak için aşağıdaki komut çalıştırılır.
````shell
python main.py
````
# Yarışmacıların Modellerini Tanımlayacakları Kısımlar
Yarışmacılar sağlanan sunucu haberleşme arayüzüne ait kodları modifiye etmeden, yanlızca ```object_detection_model.py``` içerisinde değişiklik yaparak modellerini sisteme entegre edebilirler.
Geliştirilen modellerin ``ObjectDetectionModel`` sınıfına entegrasyonu ile ilgili detaylar kod içerisinde yorumlar ile belirtilmiştir. Yarışmacılar nesne tespiti ve pozisyon kestirimi için geliştirdikleri algoritmalarını yorumlara bakarak ilgili yerlere entegre edebilirler.

## Logların İncelenmesi
Sistem çalışma esnasında gerçekleştirdiği işlemleri `_logs` klasörü altında kayıt altına almaktadır. Sistem hataları bu log dosyaları üzerinden takip edilebilir. Yarışma esnasında yapılacak itirazlarda log dosyası içeriği değerlendirmeye alınacaktır.

# Sorumluluk Beyanı
Bu sistem yarışmacılar için referans olarak sunulmuştur. Bu sistemi kullanmak kullanıcıların tercihine bağlıdır. Sağlanan sistem dolayısı ile oluşabilecek problemler yarışmacıların sorumluluğundadır.
Yarışmacı takımlar sunucu ile sağlıklı iletişim kuracak yapıyı geliştirmek/test etmek sorumluluğundadır. Teknofest, sistemde değişiklik yapma hakkını saklı tutar.
