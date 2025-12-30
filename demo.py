import streamlit as st
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd
import numpy as np
import io

# ページ設定
st.set_page_config(page_title="音声詳細分析", layout="wide")

st.title("音声詳細分析アプリ")
st.markdown("音声ファイル（WAV, MP3）をアップロードしてください。<br>※処理には数秒〜数十秒かかる場合があります。", unsafe_allow_html=True)

# --- サイドバー：設定 ---
st.sidebar.header("分析設定")

# ファイルアップロード
uploaded_file = st.sidebar.file_uploader("ファイルをアップロード", type=["wav", "mp3"])

# PDF用の情報入力
patient_id = st.sidebar.text_input("ID / ファイル名", value="test_patient")
date_str = st.sidebar.text_input("日付", value="20251231")

# 分析パラメータ
threshold = st.sidebar.slider("無音判定の閾値", 0.0, 0.5, 0.05)
fmin_val = st.sidebar.number_input("最小Hz (C1)", value=librosa.note_to_hz('C1'))
fmax_val = st.sidebar.number_input("最大Hz (C6)", value=librosa.note_to_hz('C6'))

# --- メイン処理 ---
if uploaded_file is not None:
    # 進行状況を表示
    with st.spinner('分析中...しばらくお待ちください...'):
        
        # 音声読み込み
        y, sr = librosa.load(uploaded_file, sr=None)
        
        # 基本情報
        duration = librosa.get_duration(y=y, sr=sr)
        st.success(f"読み込み完了: {duration:.2f}秒 / {sr}Hz")
        
        # 音声再生バー
        st.audio(uploaded_file)

        # --- 計算処理 (元のロジック) ---
        # スペクトログラム
        spec1 = librosa.stft(y, n_fft=512, hop_length=128)
        specdb1 = librosa.amplitude_to_db(np.abs(spec1), ref=1.0, top_db=60)
        
        spec2 = librosa.stft(y, n_fft=4096, hop_length=1024)
        specdb2 = librosa.amplitude_to_db(np.abs(spec2), ref=1.0, top_db=60)

        # ピッチ・インテンシティ
        # ※Mac等でnumbaのエラーが出ないよう例外処理を追加
        try:
            pitch, voiced_flag, voiced_probs = librosa.pyin(y, fmin=fmin_val, fmax=fmax_val)
        except Exception as e:
            st.error(f"ピッチ抽出でエラーが発生しました: {e}")
            pitch = np.full_like(y, np.nan)

        rms = librosa.feature.rms(y=y, center=True)
        rms_db = librosa.amplitude_to_db(rms, ref=1.0, top_db=60)
        
        times = librosa.times_like(pitch, sr=sr)
        
        # ゼロクロッシング
        y_clean = y.copy()
        y_clean[np.abs(y_clean) < threshold] = 0
        zcr = librosa.feature.zero_crossing_rate(y_clean)

        # --- グラフ描画 ---
        fig = plt.figure(figsize=(12, 8))
        
        # 1. 波形 & ピッチ
        plt.subplot(2, 2, 1)
        librosa.display.waveshow(y, sr=sr, axis='time', color='blue', alpha=0.5, label='Waveform')
        plt.ylabel("Amplitude")
        plt.title("Waveform + Pitch + Intensity")
        
        # 2軸目でピッチとインテンシティ
        ax2 = plt.gca().twinx()
        ax2.plot(times, pitch, label='Pitch', color='magenta', linewidth=1.5)
        ax2.plot(times, rms_db[0], label='Intensity', color='cyan', linewidth=1.5, linestyle='--')
        ax2.set_ylabel("Frequency (Hz) / Intensity (dB)")
        ax2.legend(loc='upper right')

        # 2. ゼロクロッシング
        plt.subplot(2, 2, 2)
        plt.plot(times, zcr[0], color='black')
        plt.title("Zero Crossing Rate")
        
        # 3. 広帯域スペクトログラム
        plt.subplot(2, 2, 3)
        librosa.display.specshow(specdb1, sr=sr, hop_length=128, x_axis='time', y_axis='log', cmap='jet')
        plt.title("Wideband Spectrogram")
        plt.colorbar(format='%+2.0f dB')

        # 4. 狭帯域スペクトログラム
        plt.subplot(2, 2, 4)
        librosa.display.specshow(specdb2, sr=sr, hop_length=1024, x_axis='time', y_axis='log', cmap='magma')
        plt.title("Narrowband Spectrogram")
        plt.colorbar(format='%+2.0f dB')

        plt.tight_layout()
        
        # 画面に表示
        st.pyplot(fig)

        # --- PDFダウンロード ---
        buf = io.BytesIO()
        plt.savefig(buf, format="pdf", dpi=300)
        buf.seek(0)
        
        file_name = f"{patient_id}_{date_str}.pdf"
        st.download_button(
            label="📄 PDFとして保存",
            data=buf,
            file_name=file_name,
            mime="application/pdf"
        )