from keras.models import Sequential, save_model, load_model
from keras.layers import Dense
from os import path

class YapaySinirAgi:
    model = None

    def _init_(self, model_path=None):
        if model_path is not None:
            if path.isdir(model_path):
                self.yukle(model_path)
                return
            else:
                raise "hata model dosyası yok"
        
        self.model = Sequential()
        self.model.add(Dense(784, input_dim=784,activation='relu'))
        self.model.add(Dense(120, activation='relu'))
        self.model.add(Dense(160, activation='relu'))
        self.model.add(Dense(10, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        
    def egit(self, egitimVerisi, devir =50):
        self.model.fit(egitimVerisi.x, egitimVerisi.y, epochs=devir, batch_size=64)
        _,acc = self.model.evaluate(egitimVerisi.x,egitimVerisi.y)
        print("Eğitim tamamlandı, Doğruluk:",str(acc))
        
    def kaydet(self,dosyaAd):
        save_model(self.model, dosyaAd)
    
    def yukle(self,dosyaKonum):
        self.model = load_model(dosyaKonum)