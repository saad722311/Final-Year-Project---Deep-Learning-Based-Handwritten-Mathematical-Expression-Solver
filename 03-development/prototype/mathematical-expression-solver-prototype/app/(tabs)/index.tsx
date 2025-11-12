import React, { useState, useRef } from 'react';
import {
  StyleSheet,
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  Dimensions,
  Alert,
  GestureResponderEvent,
  Image,
} from 'react-native';
import Svg, { Path } from 'react-native-svg';
import * as ImagePicker from 'expo-image-picker';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

// Type definitions
interface LLMOption {
  id: string;
  name: string;
  color: string;
}

interface Step {
  step: number;
  description: string;
  equation: string;
}

interface DummySolution {
  expression: string;
  steps: Step[];
  confidence: number;
}

interface Solution extends DummySolution {
  llmId: string;
  llmName: string;
  color: string;
}

interface Ratings {
  [key: string]: number;
}

interface PathData {
  path: string;
  color: string;
  strokeWidth: number;
}

// Star Icon Component
const StarIcon: React.FC<{ filled: boolean; color: string; size?: number }> = ({
  filled,
  color,
  size = 24,
}) => (
  <Text style={{ fontSize: size, color: filled ? color : '#D1D5DB' }}>
    {filled ? '★' : '☆'}
  </Text>
);

export default function TabOneScreen() {
  const [paths, setPaths] = useState<PathData[]>([]);
  const [currentPath, setCurrentPath] = useState<string>('');
  const [selectedLLMs, setSelectedLLMs] = useState<string[]>(['deepseek']);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [solutions, setSolutions] = useState<Solution[]>([]);
  const [eraserMode, setEraserMode] = useState<boolean>(false);
  const [showLLMPicker, setShowLLMPicker] = useState<boolean>(false);
  const [ratings, setRatings] = useState<Ratings>({});
  const [isDrawing, setIsDrawing] = useState<boolean>(false);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);

  const llmOptions: LLMOption[] = [
    { id: 'deepseek', name: 'DeepSeek R1', color: '#3B82F6' },
    { id: 'gpt', name: 'GPT-4', color: '#10B981' },
  ];

  // Dummy solutions
  const dummySolutions: { [key: string]: DummySolution } = {
    deepseek: {
      expression: '2x + 5 = 13',
      steps: [
        { step: 1, description: 'Start with the equation', equation: '2x + 5 = 13' },
        { step: 2, description: 'Subtract 5 from both sides', equation: '2x + 5 - 5 = 13 - 5' },
        { step: 3, description: 'Simplify', equation: '2x = 8' },
        { step: 4, description: 'Divide both sides by 2', equation: '2x ÷ 2 = 8 ÷ 2' },
        { step: 5, description: 'Final solution', equation: 'x = 4' },
      ],
      confidence: 0.95,
    },
    gpt: {
      expression: '2x + 5 = 13',
      steps: [
        { step: 1, description: 'Given equation', equation: '2x + 5 = 13' },
        { step: 2, description: 'Isolate the term with x by subtracting 5', equation: '2x = 13 - 5' },
        { step: 3, description: 'Evaluate the right side', equation: '2x = 8' },
        { step: 4, description: 'Solve for x by dividing by 2', equation: 'x = 8/2' },
        { step: 5, description: 'Solution', equation: 'x = 4' },
      ],
      confidence: 0.92,
    },
  };

  // Better touch handling for Apple Pencil
  const handleTouchStart = (event: GestureResponderEvent) => {
    const { locationX, locationY } = event.nativeEvent;
    setIsDrawing(true);
    setCurrentPath(`M ${locationX.toFixed(2)} ${locationY.toFixed(2)}`);
  };

  const handleTouchMove = (event: GestureResponderEvent) => {
    if (!isDrawing) return;
    const { locationX, locationY } = event.nativeEvent;
    setCurrentPath((prev) => `${prev} L ${locationX.toFixed(2)} ${locationY.toFixed(2)}`);
  };

  const handleTouchEnd = () => {
    if (isDrawing && currentPath) {
      setPaths([
        ...paths,
        {
          path: currentPath,
          color: eraserMode ? '#FFFFFF' : '#000000',
          strokeWidth: eraserMode ? 20 : 3,
        },
      ]);
      setCurrentPath('');
    }
    setIsDrawing(false);
  };

  const clearCanvas = () => {
    setPaths([]);
    setCurrentPath('');
    setUploadedImage(null);
    setSolutions([]);
    setRatings({});
  };

  const toggleLLM = (llmId: string) => {
    setSelectedLLMs((prev) => {
      if (prev.includes(llmId)) {
        return prev.filter((id) => id !== llmId);
      } else {
        return [...prev, llmId];
      }
    });
  };

  // Image picker function
  const pickImage = async () => {
    // Request permission
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    
    if (status !== 'granted') {
      Alert.alert('Permission Required', 'Sorry, we need camera roll permissions to upload images!');
      return;
    }

    // Launch image picker
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled && result.assets[0]) {
      setUploadedImage(result.assets[0].uri);
      // Clear drawn paths when uploading image
      setPaths([]);
      setCurrentPath('');
    }
  };

  // Take photo function
  const takePhoto = async () => {
    // Request permission
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    
    if (status !== 'granted') {
      Alert.alert('Permission Required', 'Sorry, we need camera permissions to take photos!');
      return;
    }

    // Launch camera
    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: true,
      quality: 1,
    });

    if (!result.canceled && result.assets[0]) {
      setUploadedImage(result.assets[0].uri);
      // Clear drawn paths when taking photo
      setPaths([]);
      setCurrentPath('');
    }
  };

  const processMath = async () => {
    if (selectedLLMs.length === 0) {
      Alert.alert('Error', 'Please select at least one LLM');
      return;
    }

    // Check if there's input (either drawing or uploaded image)
    if (paths.length === 0 && !uploadedImage) {
      Alert.alert('Error', 'Please draw an expression or upload an image');
      return;
    }

    setIsProcessing(true);
    setSolutions([]);
    setRatings({});
    setShowLLMPicker(false);

    // Simulate processing
    await new Promise((resolve) => setTimeout(resolve, 2000));

    const newSolutions: Solution[] = selectedLLMs.map((llmId) => ({
      llmId,
      llmName: llmOptions.find((l) => l.id === llmId)!.name,
      color: llmOptions.find((l) => l.id === llmId)!.color,
      ...dummySolutions[llmId],
    }));

    setSolutions(newSolutions);
    setIsProcessing(false);
  };

  const rateSolution = (llmId: string, rating: number) => {
    setRatings((prev) => ({ ...prev, [llmId]: rating }));
  };

  return (
    <View style={styles.container}>
      <ScrollView 
        style={styles.scrollView} 
        contentContainerStyle={styles.scrollContent}
        scrollEnabled={!isDrawing}
      >
        {/* Header */}
        <View style={styles.header}>
          <Text style={styles.headerTitle}>Handwritten Math Solver</Text>
          <Text style={styles.headerSubtitle}>
            Draw your mathematical expression or upload an image
          </Text>
        </View>

        {/* Canvas Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Input Canvas</Text>

          <View style={styles.canvasContainer}>
            <View
              style={styles.canvasWrapper}
              onStartShouldSetResponder={() => !uploadedImage} // Disable drawing if image uploaded
              onMoveShouldSetResponder={() => !uploadedImage}
              onResponderGrant={handleTouchStart}
              onResponderMove={handleTouchMove}
              onResponderRelease={handleTouchEnd}
              onResponderTerminate={handleTouchEnd}
            >
              {uploadedImage ? (
                // Show uploaded image
                <Image 
                  source={{ uri: uploadedImage }} 
                  style={styles.uploadedImage}
                  resizeMode="contain"
                />
              ) : (
                // Show drawing canvas
                <Svg height="400" width={SCREEN_WIDTH - 72} style={styles.svg}>
                  {paths.map((p, index) => (
                    <Path
                      key={`path-${index}`}
                      d={p.path}
                      stroke={p.color}
                      strokeWidth={p.strokeWidth}
                      fill="none"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  ))}
                  {currentPath && (
                    <Path
                      d={currentPath}
                      stroke={eraserMode ? '#FFFFFF' : '#000000'}
                      strokeWidth={eraserMode ? 20 : 3}
                      fill="none"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  )}
                </Svg>
              )}
              {isProcessing && (
                <View style={styles.processingOverlay}>
                  <Text style={styles.processingText}>Processing...</Text>
                </View>
              )}
            </View>
          </View>

          {/* Controls */}
          <View style={styles.controls}>
            {!uploadedImage ? (
              <>
                <TouchableOpacity
                  style={[styles.button, eraserMode && styles.buttonActive]}
                  onPress={() => setEraserMode(!eraserMode)}
                >
                  <Text style={[styles.buttonText, eraserMode && styles.buttonTextActive]}>
                    ✏️ Eraser
                  </Text>
                </TouchableOpacity>

                <TouchableOpacity
                  style={[styles.button, styles.buttonDanger]}
                  onPress={clearCanvas}
                >
                  <Text style={[styles.buttonText, styles.buttonTextWhite]}>🗑️ Clear</Text>
                </TouchableOpacity>
              </>
            ) : (
              <TouchableOpacity
                style={[styles.button, styles.buttonDanger, { flex: 1 }]}
                onPress={clearCanvas}
              >
                <Text style={[styles.buttonText, styles.buttonTextWhite]}>🗑️ Remove Image</Text>
              </TouchableOpacity>
            )}
          </View>

          {/* Image Upload Buttons */}
          {!uploadedImage && paths.length === 0 && (
            <View style={styles.uploadSection}>
              <Text style={styles.orText}>— OR —</Text>
              <View style={styles.uploadButtons}>
                <TouchableOpacity
                  style={[styles.button, styles.buttonUpload]}
                  onPress={pickImage}
                >
                  <Text style={styles.buttonText}>📷 Upload Image</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[styles.button, styles.buttonUpload]}
                  onPress={takePhoto}
                >
                  <Text style={styles.buttonText}>📸 Take Photo</Text>
                </TouchableOpacity>
              </View>
            </View>
          )}

          {/* LLM Selection */}
          <View style={styles.llmSection}>
            <Text style={styles.label}>Select AI Models</Text>

            <TouchableOpacity
              style={styles.llmPicker}
              onPress={() => setShowLLMPicker(!showLLMPicker)}
            >
              <Text style={styles.llmPickerText}>
                {selectedLLMs.length === 0
                  ? 'Select models...'
                  : selectedLLMs
                      .map((id) => llmOptions.find((l) => l.id === id)!.name)
                      .join(', ')}
              </Text>
            </TouchableOpacity>

            {showLLMPicker && (
              <View style={styles.llmDropdown}>
                {llmOptions.map((llm) => (
                  <TouchableOpacity
                    key={llm.id}
                    style={styles.llmOption}
                    onPress={() => toggleLLM(llm.id)}
                  >
                    <View style={styles.llmOptionContent}>
                      <Text style={styles.checkbox}>
                        {selectedLLMs.includes(llm.id) ? '☑' : '☐'}
                      </Text>
                      <View style={[styles.llmDot, { backgroundColor: llm.color }]} />
                      <Text style={styles.llmOptionText}>{llm.name}</Text>
                    </View>
                  </TouchableOpacity>
                ))}
              </View>
            )}
          </View>

          {/* Process Button */}
          <TouchableOpacity
            style={[
              styles.processButton,
              (isProcessing || selectedLLMs.length === 0 || (paths.length === 0 && !uploadedImage)) && 
              styles.processButtonDisabled,
            ]}
            onPress={processMath}
            disabled={isProcessing || selectedLLMs.length === 0 || (paths.length === 0 && !uploadedImage)}
          >
            <Text style={styles.processButtonText}>
              {isProcessing ? '⏳ Processing...' : 'Solve Expression'}
            </Text>
          </TouchableOpacity>
        </View>

        {/* Solutions Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>
            {solutions.length > 1 ? 'Comparative Solutions' : 'Solution'}
          </Text>

          {solutions.length === 0 ? (
            <View style={styles.emptyState}>
              <Text style={styles.emptyStateText}>No solutions yet</Text>
              <Text style={styles.emptyStateSubtext}>
                Draw an expression, upload an image, or take a photo
              </Text>
            </View>
          ) : (
            <View style={styles.solutionsContainer}>
              {solutions.map((solution) => (
                <View
                  key={solution.llmId}
                  style={[styles.solutionCard, { borderColor: solution.color }]}
                >
                  {/* LLM Header */}
                  <View style={styles.solutionHeader}>
                    <View style={styles.solutionHeaderLeft}>
                      <View style={[styles.llmDot, { backgroundColor: solution.color }]} />
                      <Text style={styles.llmName}>{solution.llmName}</Text>
                    </View>
                    <Text style={styles.confidence}>
                      {(solution.confidence * 100).toFixed(0)}%
                    </Text>
                  </View>

                  {/* Recognized Expression */}
                  <View style={styles.expressionBox}>
                    <Text style={styles.expressionLabel}>Recognized Expression:</Text>
                    <Text style={styles.expressionText}>{solution.expression}</Text>
                  </View>

                  {/* Solution Steps */}
                  <View style={styles.stepsContainer}>
                    {solution.steps.map((step) => (
                      <View key={step.step} style={styles.stepCard}>
                        <View style={styles.stepContent}>
                          <View
                            style={[styles.stepNumber, { backgroundColor: solution.color }]}
                          >
                            <Text style={styles.stepNumberText}>{step.step}</Text>
                          </View>
                          <View style={styles.stepTextContainer}>
                            <Text style={styles.stepDescription}>{step.description}</Text>
                            <Text style={styles.stepEquation}>{step.equation}</Text>
                          </View>
                        </View>
                      </View>
                    ))}
                  </View>

                  {/* Rating */}
                  <View style={styles.ratingContainer}>
                    <Text style={styles.ratingLabel}>Rate this solution:</Text>
                    <View style={styles.starsContainer}>
                      {[1, 2, 3, 4, 5].map((star) => (
                        <TouchableOpacity
                          key={star}
                          onPress={() => rateSolution(solution.llmId, star)}
                        >
                          <StarIcon
                            filled={ratings[solution.llmId] >= star}
                            color={solution.color}
                            size={32}
                          />
                        </TouchableOpacity>
                      ))}
                    </View>
                    {ratings[solution.llmId] && (
                      <Text style={styles.ratingText}>
                        You rated: {ratings[solution.llmId]} star
                        {ratings[solution.llmId] > 1 ? 's' : ''}
                      </Text>
                    )}
                  </View>
                </View>
              ))}
            </View>
          )}
        </View>

        {/* Feature Notes */}
        <View style={styles.footer}>
          <Text style={styles.footerTitle}>Prototype Features</Text>
          <Text style={styles.footerText}>
            ✓ Handwritten input capture with Apple Pencil support
          </Text>
          <Text style={styles.footerText}>✓ Image upload & camera capture</Text>
          <Text style={styles.footerText}>✓ Multiple LLM integration</Text>
          <Text style={styles.footerText}>✓ Step-by-step solutions</Text>
          <Text style={styles.footerText}>✓ Comparative display & ratings</Text>
          <Text style={[styles.footerText, styles.footerNote]}>
            Note: This is a front-end prototype with simulated solutions
          </Text>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F0F4FF',
  },
  scrollView: {
    flex: 1,
  },
  scrollContent: {
    paddingTop: 20,
  },
  header: {
    backgroundColor: '#FFFFFF',
    padding: 24,
    marginBottom: 16,
    borderRadius: 12,
    marginHorizontal: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 3,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#1F2937',
    marginBottom: 8,
  },
  headerSubtitle: {
    fontSize: 16,
    color: '#6B7280',
  },
  section: {
    backgroundColor: '#FFFFFF',
    padding: 20,
    marginBottom: 16,
    marginHorizontal: 16,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 3,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#1F2937',
    marginBottom: 16,
  },
  canvasContainer: {
    marginBottom: 16,
  },
  canvasWrapper: {
    borderWidth: 3,
    borderColor: '#D1D5DB',
    borderRadius: 12,
    overflow: 'hidden',
    backgroundColor: '#FFFFFF',
    height: 400,
  },
  svg: {
    backgroundColor: '#FFFFFF',
  },
  uploadedImage: {
    width: '100%',
    height: '100%',
  },
  processingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  processingText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#3B82F6',
  },
  controls: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 16,
  },
  button: {
    flex: 1,
    backgroundColor: '#E5E7EB',
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  buttonActive: {
    backgroundColor: '#F97316',
  },
  buttonDanger: {
    backgroundColor: '#EF4444',
  },
  buttonUpload: {
    backgroundColor: '#3B82F6',
  },
  buttonText: {
    fontSize: 16,
    fontWeight: '600',
    color: '#374151',
  },
  buttonTextActive: {
    color: '#FFFFFF',
  },
  buttonTextWhite: {
    color: '#FFFFFF',
  },
  uploadSection: {
    marginBottom: 16,
    alignItems: 'center',
  },
  orText: {
    fontSize: 14,
    color: '#9CA3AF',
    marginVertical: 12,
    textAlign: 'center',
  },
  uploadButtons: {
    flexDirection: 'row',
    gap: 12,
    width: '100%',
  },
  llmSection: {
    marginBottom: 16,
  },
  label: {
    fontSize: 14,
    fontWeight: '500',
    color: '#374151',
    marginBottom: 8,
  },
  llmPicker: {
    borderWidth: 2,
    borderColor: '#D1D5DB',
    borderRadius: 8,
    padding: 16,
    backgroundColor: '#FFFFFF',
  },
  llmPickerText: {
    fontSize: 16,
    color: '#374151',
  },
  llmDropdown: {
    marginTop: 8,
    borderWidth: 2,
    borderColor: '#E5E7EB',
    borderRadius: 8,
    backgroundColor: '#FFFFFF',
    overflow: 'hidden',
  },
  llmOption: {
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#F3F4F6',
  },
  llmOptionContent: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  checkbox: {
    fontSize: 20,
    color: '#3B82F6',
  },
  llmDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  llmOptionText: {
    fontSize: 16,
    color: '#374151',
  },
  processButton: {
    backgroundColor: '#3B82F6',
    paddingVertical: 16,
    borderRadius: 8,
    alignItems: 'center',
  },
  processButtonDisabled: {
    backgroundColor: '#9CA3AF',
  },
  processButtonText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#FFFFFF',
  },
  emptyState: {
    paddingVertical: 80,
    alignItems: 'center',
  },
  emptyStateText: {
    fontSize: 18,
    color: '#9CA3AF',
    marginBottom: 8,
  },
  emptyStateSubtext: {
    fontSize: 14,
    color: '#9CA3AF',
  },
  solutionsContainer: {
    gap: 16,
  },
  solutionCard: {
    borderWidth: 2,
    borderRadius: 12,
    padding: 16,
    backgroundColor: '#FFFFFF',
  },
  solutionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  solutionHeaderLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  llmName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#1F2937',
  },
  confidence: {
    fontSize: 14,
    color: '#6B7280',
  },
  expressionBox: {
    backgroundColor: '#F9FAFB',
    padding: 12,
    borderRadius: 8,
    marginBottom: 16,
  },
  expressionLabel: {
    fontSize: 12,
    color: '#6B7280',
    marginBottom: 4,
  },
  expressionText: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1F2937',
    fontFamily: 'Courier',
  },
  stepsContainer: {
    gap: 12,
    marginBottom: 16,
  },
  stepCard: {
    backgroundColor: '#FFFFFF',
    borderWidth: 1,
    borderColor: '#E5E7EB',
    borderRadius: 8,
    padding: 12,
  },
  stepContent: {
    flexDirection: 'row',
    gap: 12,
  },
  stepNumber: {
    width: 32,
    height: 32,
    borderRadius: 16,
    justifyContent: 'center',
    alignItems: 'center',
  },
  stepNumberText: {
    color: '#FFFFFF',
    fontWeight: 'bold',
    fontSize: 14,
  },
  stepTextContainer: {
    flex: 1,
  },
  stepDescription: {
    fontSize: 14,
    color: '#6B7280',
    marginBottom: 4,
  },
  stepEquation: {
    fontSize: 16,
    fontWeight: '600',
    color: '#1F2937',
    fontFamily: 'Courier',
  },
  ratingContainer: {
    borderTopWidth: 1,
    borderTopColor: '#E5E7EB',
    paddingTop: 16,
  },
  ratingLabel: {
    fontSize: 14,
    color: '#6B7280',
    marginBottom: 8,
  },
  starsContainer: {
    flexDirection: 'row',
    gap: 8,
  },
  ratingText: {
    fontSize: 14,
    color: '#6B7280',
    marginTop: 8,
  },
  footer: {
    backgroundColor: '#FFFFFF',
    padding: 20,
    marginBottom: 32,
    marginHorizontal: 16,
    borderRadius: 12,
  },
  footerTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1F2937',
    marginBottom: 12,
  },
  footerText: {
    fontSize: 14,
    color: '#6B7280',
    marginBottom: 6,
  },
  footerNote: {
    marginTop: 8,
    fontStyle: 'italic',
  },
});