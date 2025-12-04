import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers

# 0) CSV 불러오기
file_path = r"C:\Users\jinhu\OneDrive\문서\20222114_신진협_기계학습 튜닝 및 모델 변경\서울시 인터넷 쇼핑몰 소비자 피해 상담 정보.csv"
df = pd.read_csv(file_path, encoding="cp949")

# 1) 피해유형 있는 행만 필터링
df_clean = df.dropna(subset=["피해유형"])

# 2) 학습 텍스트 생성 (물품분류 + 구매유형)
df_clean["text"] = df_clean["물품분류"].astype(str) + " " + df_clean["구매유형"].astype(str)

texts = df_clean["text"].tolist()                     # 문자열 리스트
labels_cat = df_clean["피해유형"].astype("category")  # 범주형

# 라벨 → 정수 변환
labels = labels_cat.cat.codes.to_numpy()
label_names = list(labels_cat.cat.categories)

# 3) train/test 분리
X_train_full, X_test, y_train_full, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# 4) train/validation 분리
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
)

# 5) tf.data.Dataset으로 만들기
BATCH_SIZE = 32

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE)
val_ds   = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
test_ds  = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

# 6) 텍스트 벡터화 및 adapt
vectorize = layers.TextVectorization(
    max_tokens=10000,
    output_sequence_length=50
)

vectorize.adapt(X_train)

# 7) 튜닝된 모델 정의
num_classes = len(label_names)

model = tf.keras.Sequential([
    vectorize,                             # 문자열 → 정수 시퀀스
    layers.Embedding(10000, 128),          # 64 → 128 차원
    layers.GlobalAveragePooling1D(),
    layers.Dense(128, activation="relu"),  # 은닉층 확대
    layers.Dropout(0.5),                   # Dropout 추가
    layers.Dense(64, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 8) summary (더미 입력으로 build)
dummy_input = tf.constant(["임시 문장"])
_ = model(dummy_input)

print("\n===== TUNED MODEL SUMMARY =====\n")
model.summary()

# 9) EarlyStopping 콜백 정의
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

# 10) 모델 학습 (튜닝 버전)
history = model.fit(
    train_ds,
    epochs=10,              # 5 → 10
    validation_data=val_ds,
    callbacks=[early_stop]
)

# 11) 검증 정확도 출력
final_val_acc = history.history["val_accuracy"][-1]
print("\n튜닝 모델 최종 검증 정확도:", final_val_acc)

# 12) 테스트 정확도 출력
test_loss, test_acc = model.evaluate(test_ds)
print("\n튜닝 모델 테스트 정확도:", test_acc)

# 13) 학습 히스토리를 CSV로 저장 (그래프/엑셀용)
hist_df = pd.DataFrame(history.history)
hist_df.index = hist_df.index + 1  # epoch 번호를 1부터 보이게 하려면:
hist_df.index.name = "epoch"
hist_df.to_csv("history.csv", encoding="utf-8-sig")
print("\n튜닝 모델 학습 기록을 history.csv 로 저장했습니다.")