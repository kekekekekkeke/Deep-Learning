## Introduction

RNN (Recurrent Neural Network) modelleri, sıralı verilerdeki zaman bağımlılıklarını yakalamak için yaygın olarak kullanılır. 
Bu çalışmada, aynı metin sınıflandırma görevi üzerinde iki farklı RNN implementasyonu (NumPy ile el ile yazılmış ve PyTorch tabanlı) karşılaştırılmıştır. 
Amaç, her iki yaklaşımın kod yapısı, performans, esneklik ve eğitim takibi açısından güçlü ve zayıf yönlerini ortaya koymaktır.

## Methods

### NumPy Implementasyonu

* **Model Mimari**: Tek katmanlı temel RNN hücresi
* **İleri Geçiş**: `h_t = p.tanh(W_x x_t + W_h h_{t-1} + b)`
* **Geri Yayılım**: El ile türev hesaplamaları ve ağırlık güncellemeleri
* **Optimizasyon**: Stokastik gradyan inişi (SGD)
* **Veri Hazırlama**: Dize öbekleme, one-hot kodlama

### PyTorch Implementasyonu

* **Model Mimari**: `nn.RNN` katmanı kullanılarak tek katmanlı RNN
* **İleri Geçiş ve Geri Yayılım**: Otomatik türev (`autograd`)
* **Optimizasyon**: `torch.optim.Adam`
* **Kayıp Fonksiyonu**: `nn.CrossEntropyLoss`
* **Veri Yükleme**: `Dataset` ve `DataLoader`

Her iki model de aynı eğitim ve test veri seti (10.000 eğitim örneği, %80 eğitim/%20 test) ile çalıştırılmıştır. Eğitim 20 epoch boyunca gerçekleştirilmiştir.

## Results

 Eğitim Doğruluğu    NumPy RNN =(%)  87.0   PyTorch RNN  = (%)  100.0  
 

## Discussion

NumPy implementasyonu, RNN’in iç işleyişini öğrenmek ve algoritmik detaylara hakim olmak için etkili bir araçtır. Ancak, el ile yazılan geri yayılım kodu karmaşık ve hata yapmaya açıktır. Büyük veri setlerinde ve uzun paralellik gerektiren uygulamalarda performans yetersiz kalmaktadır.

Buna karşılık PyTorch:

* Otomatik türev, modüler yapı ve geniş API desteği sağlar.
* GPU desteği ve optimize edilmiş C++/CUDA backend ile eğitim süresini ciddi oranda azaltır.
* `DataLoader`, `TensorBoard` gibi araçlarla eğitim takibi ve hata analizi kolaylaşır.

## Conclusion

NumPy temelli RNN, eğitim amaçlı iç mekanizma öğrenimi için uygundur; ancak gerçek dünya uygulamalarında PyTorch tabanlı modeller, hız, ölçeklenebilirlik ve kod bakım kolaylığı bakımından daha avantajlıdır.
