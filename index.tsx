import * as ImageManipulator from 'expo-image-manipulator';
import * as ImagePicker from 'expo-image-picker';
import { toByteArray } from 'base64-js';
import { decode as decodeJpeg } from 'jpeg-js';
import React, { useEffect, useState } from 'react';
import { ActivityIndicator, Alert, Image, StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { loadTensorflowModel, type TfliteModel } from 'react-native-fast-tflite';

// eslint-disable-next-line @typescript-eslint/no-require-imports
const CLASSIFIER_MODEL_FILE = require('../../assets/mushroom_model_lite.tflite');

const CLASSIFIER_IMAGE_SIZE = 224;
const CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.55;
const EDIBLE_CONFIDENCE_THRESHOLD = 0.65;
const POISONOUS_CONFIDENCE_THRESHOLD = 0.6;
const NON_MUSHROOM_CONFIDENCE_THRESHOLD = 0.55;
const NON_MUSHROOM_LABEL = 'mantar_degil';
const LABELS = [NON_MUSHROOM_LABEL, 'Yenilebilir', 'Zehirli'];
const IMAGE_PICKER_OPTIONS: ImagePicker.ImagePickerOptions = {
  mediaTypes: ImagePicker.MediaTypeOptions.Images,
  allowsEditing: true,
  aspect: [1, 1],
  quality: 1,
};

const toArrayBuffer = (typedArray: Float32Array | Uint8Array | Int8Array) =>
  typedArray.buffer.slice(
    typedArray.byteOffset,
    typedArray.byteOffset + typedArray.byteLength
  ) as ArrayBuffer;

const getSelectedImageUri = (response: ImagePicker.ImagePickerResult | null | undefined) => {
  if (!response || response.canceled) {
    return null;
  }

  const asset = response.assets?.[0];

  return asset?.uri ?? null;
};

const prepareImageInput = async (uri: string, model: TfliteModel, width: number, height: number) => {
  const resizedImage = await ImageManipulator.manipulateAsync(
    uri,
    [{ resize: { width, height } }],
    { base64: true, compress: 1, format: ImageManipulator.SaveFormat.JPEG }
  );

  if (!resizedImage.base64) {
    throw new Error('IMAGE_BASE64_NOT_CREATED');
  }

  const jpegBytes = toByteArray(resizedImage.base64);
  const decodedImage = decodeJpeg(jpegBytes, { useTArray: true });
  const inputType = model.inputs[0]?.dataType ?? 'float32';
  const pixelCount = width * height;

  if (inputType === 'uint8' || inputType === 'int8') {
    const input = inputType === 'int8' ? new Int8Array(pixelCount * 3) : new Uint8Array(pixelCount * 3);

    for (let pixelIndex = 0; pixelIndex < pixelCount; pixelIndex += 1) {
      const rgbaIndex = pixelIndex * 4;
      const rgbIndex = pixelIndex * 3;
      input[rgbIndex] = inputType === 'int8' ? decodedImage.data[rgbaIndex] - 128 : decodedImage.data[rgbaIndex];
      input[rgbIndex + 1] = inputType === 'int8' ? decodedImage.data[rgbaIndex + 1] - 128 : decodedImage.data[rgbaIndex + 1];
      input[rgbIndex + 2] = inputType === 'int8' ? decodedImage.data[rgbaIndex + 2] - 128 : decodedImage.data[rgbaIndex + 2];
    }

    return toArrayBuffer(input);
  }

  const input = new Float32Array(pixelCount * 3);

  for (let pixelIndex = 0; pixelIndex < pixelCount; pixelIndex += 1) {
    const rgbaIndex = pixelIndex * 4;
    const rgbIndex = pixelIndex * 3;
    input[rgbIndex] = decodedImage.data[rgbaIndex] / 255;
    input[rgbIndex + 1] = decodedImage.data[rgbaIndex + 1] / 255;
    input[rgbIndex + 2] = decodedImage.data[rgbaIndex + 2] / 255;
  }

  return toArrayBuffer(input);
};

const tensorToScores = (outputBuffer: ArrayBuffer, outputType: string) =>
  outputType === 'uint8'
    ? Array.from(new Uint8Array(outputBuffer), (value) => value / 255)
    : outputType === 'int8'
      ? Array.from(new Int8Array(outputBuffer), (value) => (value + 128) / 255)
    : Array.from(new Float32Array(outputBuffer));

const softmax = (values: number[]) => {
  const maxValue = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - maxValue));
  const sum = exps.reduce((total, value) => total + value, 0);

  return exps.map((value) => value / sum);
};

const normalizeScores = (scores: number[]) => {
  const sum = scores.reduce((total, score) => total + score, 0);
  const looksLikeProbabilities =
    scores.every((score) => score >= 0 && score <= 1) && Math.abs(sum - 1) < 0.05;

  return looksLikeProbabilities ? scores : softmax(scores);
};

const parseClassification = (outputBuffer: ArrayBuffer, classifier: TfliteModel) => {
  const outputType = classifier.outputs[0]?.dataType ?? 'float32';
  const scores = tensorToScores(outputBuffer, outputType);

  if (scores.length === 1) {
    const poisonousScore = scores[0];
    const edibleScore = 1 - poisonousScore;
    const confidence = Math.max(poisonousScore, edibleScore);
    const label = poisonousScore >= 0.5 ? 'Zehirli' : 'Yenilebilir';

    return { label, confidence };
  }

  const probabilities = normalizeScores(scores);
  const topIndex = probabilities.reduce(
    (bestIndex, score, index) => (score > probabilities[bestIndex] ? index : bestIndex),
    0
  );

  return {
    label: LABELS[topIndex] ?? `Sınıf ${topIndex + 1}`,
    confidence: probabilities[topIndex],
  };
};

const formatDebugConfidence = (classifierConfidence: number) =>
  `Xception: %${(classifierConfidence * 100).toFixed(1)}`;

export default function App() {
  const [image, setImage] = useState<string | null>(null);
  const [result, setResult] = useState('TFLite modeli yükleniyor...');
  const [loading, setLoading] = useState(false);
  const [classifierModel, setClassifierModel] = useState<TfliteModel | null>(null);
  const [modelReady, setModelReady] = useState(false);

  useEffect(() => {
    let isMounted = true;

    const loadModel = async () => {
      try {
        const loadedClassifier = await loadTensorflowModel(CLASSIFIER_MODEL_FILE, []);

        if (isMounted) {
          setClassifierModel(loadedClassifier);
          setModelReady(true);
          setResult('Model hazır. Fotoğraf seçebilir veya çekebilirsiniz.');
        }
      } catch (error) {
        console.error(error);
        if (isMounted) {
          setResult('TFLite modeli yüklenemedi. Development build/APK ile çalıştırın.');
        }
      }
    };

    loadModel();

    return () => {
      isMounted = false;
    };
  }, []);

  const analyzeMushroom = async (uri: string) => {
    if (!classifierModel) {
      Alert.alert('Model Hazır Değil', 'Lütfen TFLite modelinin yüklenmesini bekleyin.');
      return;
    }

    setLoading(true);
    setResult('Görüntü işleniyor...');

    try {
      const classifierInput = await prepareImageInput(
        uri,
        classifierModel,
        CLASSIFIER_IMAGE_SIZE,
        CLASSIFIER_IMAGE_SIZE
      );
      const classifierOutputs = classifierModel.runSync([classifierInput]);
      const prediction = parseClassification(classifierOutputs[0], classifierModel);
      const debugConfidence = formatDebugConfidence(prediction.confidence);

      if (prediction.label === NON_MUSHROOM_LABEL) {
        if (prediction.confidence < NON_MUSHROOM_CONFIDENCE_THRESHOLD) {
          setResult(`Emin değilim (${debugConfidence})`);
          return;
        }

        setResult(`Mantar tespit edilemedi (${debugConfidence})`);
        return;
      }

      const requiredConfidence =
        prediction.label === 'Yenilebilir' ? EDIBLE_CONFIDENCE_THRESHOLD : POISONOUS_CONFIDENCE_THRESHOLD;

      if (prediction.confidence < CLASSIFICATION_CONFIDENCE_THRESHOLD || prediction.confidence < requiredConfidence) {
        setResult(`Emin değilim (${debugConfidence})`);
        return;
      }

      setResult(`${prediction.label} (${debugConfidence})`);
    } catch (error) {
      console.error(error);
      setResult('Analiz yapılamadı. Lütfen başka bir fotoğraf deneyin.');
    } finally {
      setLoading(false);
    }
  };

  const takePhoto = async () => {
    try {
      const { status } = await ImagePicker.requestCameraPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('İzin Gerekli', 'Kamerayı kullanabilmek için ayarlardan izin vermelisiniz.');
        return;
      }

      const response = await ImagePicker.launchCameraAsync(IMAGE_PICKER_OPTIONS);
      const uri = getSelectedImageUri(response);

      if (!uri) {
        setResult('Fotoğraf seçilmedi.');
        return;
      }

      setImage(uri);
      await analyzeMushroom(uri);
    } catch (error) {
      console.error(error);
      setResult('Fotoğraf alınamadı.');
    }
  };

  const pickImage = async () => {
    try {
      const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (status !== 'granted') {
        Alert.alert('İzin Gerekli', 'Galeriye erişmek için ayarlardan izin vermelisiniz.');
        return;
      }

      const response = await ImagePicker.launchImageLibraryAsync(IMAGE_PICKER_OPTIONS);
      const uri = getSelectedImageUri(response);

      if (!uri) {
        setResult('Fotoğraf seçilmedi.');
        return;
      }

      setImage(uri);
      await analyzeMushroom(uri);
    } catch (error) {
      console.error(error);
      setResult('Fotoğraf alınamadı.');
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Mantar Tanıma Sistemi</Text>

      <View style={styles.imageContainer}>
        {image ? (
          <Image source={{ uri: image }} style={styles.image} />
        ) : (
          <Text style={styles.placeholderText}>Analiz için bir fotoğraf seçin veya çekin</Text>
        )}
      </View>

      <View style={styles.resultBox}>
        {(loading || !modelReady) && <ActivityIndicator size="small" color="#e67e22" style={{ marginBottom: 10 }} />}
        <Text
          style={[
            styles.resultText,
            {
              color: result.includes('Zehirli')
                ? '#e74c3c'
                : result.includes('Yenilebilir')
                  ? '#2ecc71'
                  : '#7f8c8d',
            },
          ]}>
          {result}
        </Text>
      </View>

      <TouchableOpacity
        style={[styles.btnAction, (!modelReady || loading) && styles.btnDisabled]}
        onPress={takePhoto}
        disabled={!modelReady || loading}>
        <Text style={styles.btnText}>FOTOĞRAF ÇEK</Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={[styles.btnAction, styles.galleryButton, (!modelReady || loading) && styles.btnDisabled]}
        onPress={pickImage}
        disabled={!modelReady || loading}>
        <Text style={styles.btnText}>GALERİDEN SEÇ</Text>
      </TouchableOpacity>

      <Text style={styles.university}>Zonguldak Bülent Ecevit Üniversitesi - 2026</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fdfdfd', alignItems: 'center', justifyContent: 'center', padding: 25 },
  title: { fontSize: 28, fontWeight: 'bold', color: '#2c3e50', marginBottom: 35 },
  imageContainer: { width: 310, height: 310, backgroundColor: '#ecf0f1', borderRadius: 30, justifyContent: 'center', alignItems: 'center', marginBottom: 25, overflow: 'hidden', borderWidth: 2, borderColor: '#bdc3c7', elevation: 3 },
  image: { width: '100%', height: '100%' },
  placeholderText: { color: '#95a5a6', fontStyle: 'italic', padding: 20, textAlign: 'center' },
  resultBox: { width: '100%', padding: 20, backgroundColor: '#fff', borderRadius: 20, marginBottom: 30, alignItems: 'center', shadowColor: '#000', shadowOffset: { width: 0, height: 2 }, shadowOpacity: 0.1, shadowRadius: 5, elevation: 4 },
  resultText: { fontSize: 20, fontWeight: 'bold', textAlign: 'center', lineHeight: 28 },
  btnAction: { width: '100%', height: 65, backgroundColor: '#d35400', borderRadius: 18, justifyContent: 'center', alignItems: 'center', marginBottom: 15, shadowOpacity: 0.2, elevation: 3 },
  galleryButton: { backgroundColor: '#34495e' },
  btnDisabled: { opacity: 0.55 },
  btnText: { color: '#fff', fontSize: 18, fontWeight: 'bold', letterSpacing: 1 },
  university: { marginTop: 25, fontSize: 13, color: '#bdc3c7', fontWeight: '500' },
});
