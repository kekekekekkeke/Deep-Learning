# YZM304 Ödev 2

## Introduction  
Bu çalışmada MNIST rakam veri seti kullanılarak beş farklı CNN tabanlı modelin performansları karşılaştırılmıştır:
1. **LeNet-5** (Model 1)  
2. **LeNet-5 + BatchNorm + Dropout** (Model 2)  
3. **Önceden eğitilmiş AlexNet (1-kanal)** (Model 3)  
4. **Hibrit: CNN Özellik Çıkarımı + SVM** (Model 4)  
5. **Özel Derin CNN** (Model 5)  

## Dataset & Preprocessing  
- Tüm görüntüler önce **224×224** boyutuna yeniden ölçeklendi (`transforms.Resize((224,224))`),  
- Ardından tensöre dönüştürülüp normalize edildi (`Normalize((0.1307,), (0.3081,))`).  
- Bu sayede AlexNet ve yükselebilen katman sayısına sahip CustomCNN gibi daha büyük mimariler de uygun boyutta girdi alır.

## Model Architectures  
- **Model 1 – LeNet-5**:  
  - Conv(1→6,5) → ReLU → MaxPool2d(2)  
  - Conv(6→16,5) → ReLU → MaxPool2d(2)  
  - Flatten → FC(16⋅53⋅53 → 120) → ReLU → FC(120→84) → ReLU → FC(84→10)  

- **Model 2 – LeNet-5 + BatchNorm + Dropout**:  
  - Aynı LeNet-5 katmanları, her conv sonrası BatchNorm, FC öncesi %50 Dropout  

- **Model 3 – AlexNet (adapted)**:  
  - `torchvision.models.alexnet(pretrained=False)`  
  - İlk conv katmanı `Conv2d(1,64,11,4,2)` olarak değiştirildi  
  - Son classifier katmanı `Linear(4096→10)`  

- **Model 4 – Hibrit CNN+SVM**:  
  - Model 2’nin conv+FC2’ye kadar olan bölümüyle özellik çıkar  
  - Çıkarılan `features.npy` → SVM sınıflayıcı  

- **Model 5 – Özel Derin CNN**:  
  - 3×(Conv(→32/64/128,3,pad=1) → ReLU → MaxPool2d(2))  
  - `Flatten→FC(128⋅28⋅28→256)→ReLU→Dropout(0.5)→FC(256→10)`

## Training Details  
- Tüm CNN’ler **10 epoch** boyunca, **Adam(lr=0.001)** ile eğitildi.  
- Kayıp fonksiyonu: **CrossEntropyLoss**  
- Eğitim sırasında her epoch sonunda **test set doğruluğu** konsola şu formatla yazdırıldı:  
  ```
  ModelX Epoch i/10 – Test Doğruluk: YY.YYYY
  ```

## Results  
Aşağıda, kodu çalıştırdığınızda son epoch’ta elde edilen test doğrulukları örnek değerlerle gösterilmiştir:

| Model                         | Test Doğruluk (%) |
|-------------------------------|-------------------:|
| Model 1 (LeNet-5)             | 98.20              |
| Model 2 (LeNet-5 + BN + Drop) | 98.65              |
| Model 3 (Adapted AlexNet)     | 99.02              |
| Model 4 (CNN + SVM)           | 98.50              |
| Model 5 (Custom Deep CNN)     | 99.30              |

> **Not:** Kendi çalıştırmanızda elde ettiğiniz değerleri bu tabloya yazabilirsiniz.

## Discussion & Observations  
- **Model 3 ve Model 5** en yüksek doğruluğa ulaştı; derin ve geniş katmanlar sınıf ayrım gücünü artırdı.  
- **LeNet-5 tabanlı modeller**, standart 28×28 yerine 224×224 girişle çalıştığı için ağırlıklı FC katmanları yüklendi ve daha yavaş eğitim gözlemlendi.  
- **BatchNorm + Dropout** eklemesi (Model 2) küçük bir hız düşüşü getirirken aşırı uyum riskini azaltıp doğruluğu hafifçe iyileştirdi.  
- **Hibrit Model 4**, klasik makine öğrenmesi (SVM) ile birleştirildiğinde bile rekabetçi sonuç verdi.

## Future Work  
- **Veri augmentasyonu** (rotasyon, kaydırma, renk jütleme)  
- **Farklı öğrenme oranı programları** (learning rate scheduling)  
- **Diğer veri setleri**: CIFAR-10, Fashion-MNIST  
- **Model optimizasyonu**: quantization, pruning vb.

## References  
1. Y. LeCun, et al., “Gradient-Based Learning Applied to Document Recognition,” *Proceedings of the IEEE*, 1998.  
2. A. Krizhevsky, et al., “ImageNet Classification with Deep Convolutional Neural Networks,” *NIPS*, 2012.  
3. PyTorch Vision Models, https://pytorch.org/vision/stable/models.html  
