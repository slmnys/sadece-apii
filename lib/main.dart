import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'dart:convert';
import 'dart:io';
import 'package:http_parser/http_parser.dart';
import 'package:mime/mime.dart'; // Eklendi


import 'dart:typed_data';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Leaf Classification',
      theme: ThemeData(
        primarySwatch: Colors.green,
        visualDensity: VisualDensity.adaptivePlatformDensity,
        appBarTheme: AppBarTheme(
          backgroundColor: Colors.green[700],
          foregroundColor: Colors.white,
          elevation: 2,
        ),
      ),
      home: LeafClassificationScreen(),
    );
  }
}

class LeafClassificationScreen extends StatefulWidget {
  @override
  _LeafClassificationScreenState createState() => _LeafClassificationScreenState();
}

class _LeafClassificationScreenState extends State<LeafClassificationScreen> {
  final ImagePicker _picker = ImagePicker();
  File? _selectedImage;
  bool _isLoading = false;
  int _currentAttempt = 0;
  Map<String, dynamic>? _classificationResult;
  Uint8List? _segmentedImageBytes;

  // Android emülatör için: http://10.0.2.2:5000
  // Gerçek cihaz için: http://192.168.x.x:5000 (kendi bilgisayarınızın IP'si)
  final String serverUrl = 'https://yaprak-tani-api.onrender.com';
  final List<String> supportedClasses = ['apple', 'grape', 'orange', 'soybean', 'tomato'];

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? image = await _picker.pickImage(
        source: source,
        maxWidth: 1024,
        maxHeight: 1024,
        imageQuality: 85,
      );
      if (image != null) {
        setState(() {
          _selectedImage = File(image.path);
          _classificationResult = null;
          _segmentedImageBytes = null;
        });
      }
    } catch (e) {
      _showErrorDialog('Görsel seçme hatası: $e');
    }
  }

  Future<void> _classifyLeaf() async {
    if (_selectedImage == null) {
      _showErrorDialog('Lütfen önce bir görsel seçin');
      return;
    }

    setState(() {
      _isLoading = true;
      _currentAttempt = 0;
    });

    int maxRetries = 3;
    for (int attempt = 1; attempt <= maxRetries; attempt++) {
      setState(() {
        _currentAttempt = attempt;
      });
      
      try {
        print('Deneme $attempt/$maxRetries');
        
        // Dosyayı önce byte'lara çevir (emülatör için daha stabil)
        final bytes = await _selectedImage!.readAsBytes();
        
        var request = http.MultipartRequest('POST', Uri.parse('$serverUrl/classify'));
        
        // fromPath yerine fromBytes kullan
        request.files.add(
          http.MultipartFile.fromBytes(
            'image',
            bytes,
            filename: 'image.jpg',
            contentType: MediaType('image', 'jpeg'),
          ),
        );

        request.headers['Connection'] = 'close';
        request.headers['User-Agent'] = 'Flutter/Android';

        var response = await request.send().timeout(const Duration(seconds: 30));
        final responseData = await http.Response.fromStream(response);

        print('Response Status: ${response.statusCode}');
        print('Response Headers: ${response.headers}');
        print('Gelen cevap boyutu: ${responseData.body.length} karakter');

        if (response.statusCode == 200) {
          try {
            var jsonResponse = json.decode(responseData.body);
            if (jsonResponse['status'] == 'success') {
              setState(() {
                _classificationResult = jsonResponse['prediction'];
                if (jsonResponse['segmented_image'] != null) {
                  _segmentedImageBytes = base64Decode(jsonResponse['segmented_image']);
                }
              });
              print('✅ Başarılı! Deneme $attempt/$maxRetries');
              break; // Başarılıysa döngüden çık
            } else {
              _showErrorDialog('Sınıflandırma başarısız: ${jsonResponse['message'] ?? 'Bilinmeyen hata'}');
              break; // Backend hatası varsa tekrar deneme
            }
          } catch (jsonError) {
            if (attempt == maxRetries) {
              _showErrorDialog('JSON parse hatası: $jsonError\nCevap: ${responseData.body.substring(0, 500)}...');
            } else {
              print('JSON parse hatası, tekrar deneniyor... ($attempt/$maxRetries)');
              await Future.delayed(Duration(milliseconds: 500));
            }
          }
        } else {
          if (attempt == maxRetries) {
            _showErrorDialog('Sunucu hatası sorun burda: ${response.statusCode}\nCevap: ${responseData.body}');
          } else {
            print('HTTP ${response.statusCode} hatası, tekrar deneniyor... ($attempt/$maxRetries)');
            await Future.delayed(Duration(milliseconds: 500));
          }
        }
      } catch (e) {
        if (attempt == maxRetries) {
          _showErrorDialog('Ağ hatası (3 deneme sonrası): $e');
        } else {
          print('Ağ hatası, tekrar deneniyor... ($attempt/$maxRetries): $e');
          await Future.delayed(Duration(milliseconds: 1000));
        }
      }
    }

    setState(() {
      _isLoading = false;
    });
  }

  void _showErrorDialog(String message) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text('Hata'),
          content: Text(message),
          actions: [
            TextButton(
              child: Text('Tamam'),
              onPressed: () => Navigator.of(context).pop(),
            ),
          ],
        );
      },
    );
  }

  Color _getConfidenceColor(double confidence) {
    if (confidence >= 0.8) return Colors.green;
    if (confidence >= 0.6) return Colors.orange;
    return Colors.red;
  }

  String _getConfidenceText(double confidence) {
    if (confidence >= 0.8) return 'Yüksek Güven';
    if (confidence >= 0.6) return 'Orta Güven';
    return 'Düşük Güven';
  }

  IconData _getLeafIcon(String leafType) {
    switch (leafType.toLowerCase()) {
      case 'apple':
        return Icons.apple;
      case 'grape':
        return Icons.spa;
      case 'orange':
        return Icons.circle;
      case 'soybean':
        return Icons.grain;
      case 'tomato':
        return Icons.local_florist;
      default:
        return Icons.eco;
    }
  }

  String _getLeafDisplayName(String leafType) {
    switch (leafType.toLowerCase()) {
      case 'apple':
        return 'Elma';
      case 'grape':
        return 'Üzüm';
      case 'orange':
        return 'Portakal';
      case 'soybean':
        return 'Soya Fasulyesi';
      case 'tomato':
        return 'Domates';
      default:
        return leafType.toUpperCase();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Yaprak Sınıflandırma'),
        backgroundColor: Colors.green[700],
        foregroundColor: Colors.white,
      ),
      body: SingleChildScrollView(
        padding: EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Görsel Seçim Bölümü
            Card(
              elevation: 4,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              child: Padding(
                padding: EdgeInsets.all(16.0),
                child: Column(
                  children: [
                    Icon(Icons.photo_camera, size: 48, color: Colors.green[700]),
                    SizedBox(height: 8),
                    Text(
                      'Yaprak Görseli Seçin',
                      style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    SizedBox(height: 8),
                    Text(
                      'Sadece tek bir yaprağın olduğu bir görsel seçin',
                      style: TextStyle(
                        fontSize: 14,
                        color: Colors.grey[600],
                        fontStyle: FontStyle.italic,
                      ),
                      textAlign: TextAlign.center,
                    ),
                    SizedBox(height: 16),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: [
                        Expanded(
                          child: ElevatedButton.icon(
                            onPressed: () => _pickImage(ImageSource.camera),
                            icon: Icon(Icons.camera_alt),
                            label: Text('Kamera'),
                            style: ElevatedButton.styleFrom(
                              padding: EdgeInsets.symmetric(vertical: 12),
                            ),
                          ),
                        ),
                        SizedBox(width: 12),
                        Expanded(
                          child: ElevatedButton.icon(
                            onPressed: () => _pickImage(ImageSource.gallery),
                            icon: Icon(Icons.photo_library),
                            label: Text('Galeri'),
                            style: ElevatedButton.styleFrom(
                              padding: EdgeInsets.symmetric(vertical: 12),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
            
            SizedBox(height: 16),
            
            // Seçilen Görsel
            if (_selectedImage != null)
              Card(
                elevation: 4,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                child: Padding(
                  padding: EdgeInsets.all(16.0),
                  child: Column(
                    children: [
                      Text(
                        'Seçilen Görsel',
                        style: Theme.of(context).textTheme.titleLarge?.copyWith(
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      SizedBox(height: 12),
                      ClipRRect(
                        borderRadius: BorderRadius.circular(8),
                        child: Image.file(
                          _selectedImage!,
                          height: 200,
                          width: double.infinity,
                          fit: BoxFit.cover,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            
            SizedBox(height: 16),
            
            // Analiz Butonları
            if (_selectedImage != null)
              Card(
                elevation: 4,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                child: Padding(
                  padding: EdgeInsets.all(16.0),
                  child: Column(
                    children: [
                      Icon(Icons.analytics, size: 32, color: Colors.blue[700]),
                      SizedBox(height: 8),
                      Text(
                        'Yaprak Analizi',
                        style: Theme.of(context).textTheme.titleLarge?.copyWith(
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      SizedBox(height: 16),
                      ElevatedButton.icon(
                        onPressed: _isLoading ? null : _classifyLeaf,
                        icon: Icon(Icons.science),
                        label: Text('Yaprak Analizi'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.green[700],
                          foregroundColor: Colors.white,
                          padding: EdgeInsets.symmetric(vertical: 12, horizontal: 24),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            
            SizedBox(height: 16),
            
            // Yükleme Göstergesi
            if (_isLoading)
              Card(
                elevation: 4,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                child: Padding(
                  padding: EdgeInsets.all(32.0),
                  child: Column(
                    children: [
                      CircularProgressIndicator(
                        valueColor: AlwaysStoppedAnimation<Color>(Colors.green[700]!),
                      ),
                      SizedBox(height: 16),
                      Text(
                        'Yaprak analiz ediliyor...',
                        style: TextStyle(fontSize: 16, fontWeight: FontWeight.w500),
                      ),
                      if (_currentAttempt > 0)
                        Padding(
                          padding: EdgeInsets.only(top: 8),
                          child: Text(
                            'Deneme $_currentAttempt/3',
                            style: TextStyle(fontSize: 14, color: Colors.grey[600]),
                          ),
                        ),
                    ],
                  ),
                ),
              ),
            
            // Segmentli Görsel
            if (_segmentedImageBytes != null)
              Card(
                elevation: 4,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                child: Padding(
                  padding: EdgeInsets.all(16.0),
                  child: Column(
                    children: [
                      Text(
                        'Segmentli Yaprak',
                        style: Theme.of(context).textTheme.titleLarge?.copyWith(
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      SizedBox(height: 12),
                      ClipRRect(
                        borderRadius: BorderRadius.circular(8),
                        child: Image.memory(
                          _segmentedImageBytes!,
                          height: 200,
                          width: double.infinity,
                          fit: BoxFit.cover,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            
            // Sınıflandırma Sonuçları
            if (_classificationResult != null)
              Card(
                elevation: 4,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                child: Padding(
                  padding: EdgeInsets.all(16.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Icon(Icons.search, color: Colors.green[700]),
                          SizedBox(width: 8),
                          Text(
                            'Sınıflandırma Sonucu',
                            style: Theme.of(context).textTheme.titleLarge?.copyWith(
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                      SizedBox(height: 20),
                      
                      // Ana Tahmin
                      Container(
                        width: double.infinity,
                        padding: EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            colors: [
                              _getConfidenceColor(_classificationResult!['confidence']).withOpacity(0.1),
                              _getConfidenceColor(_classificationResult!['confidence']).withOpacity(0.05),
                            ],
                          ),
                          border: Border.all(
                            color: _getConfidenceColor(_classificationResult!['confidence']),
                            width: 2,
                          ),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Column(
                          children: [
                            Icon(
                              _getLeafIcon(_classificationResult!['predicted_class']),
                              size: 48,
                              color: _getConfidenceColor(_classificationResult!['confidence']),
                            ),
                            SizedBox(height: 12),
                            Text(
                              _getLeafDisplayName(_classificationResult!['predicted_class']),
                              style: TextStyle(
                                fontSize: 24,
                                fontWeight: FontWeight.bold,
                                color: _getConfidenceColor(_classificationResult!['confidence']),
                              ),
                            ),
                            SizedBox(height: 8),
                            Text(
                              '${(_classificationResult!['confidence_percentage']).toStringAsFixed(1)}% - ${_getConfidenceText(_classificationResult!['confidence'])}',
                              style: TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.w500,
                                color: Colors.grey[600],
                              ),
                            ),
                          ],
                        ),
                      ),
                      
                      SizedBox(height: 20),
                      
                      // Tüm Olasılıklar
                      Text(
                        'Tüm Sınıf Olasılıkları',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      SizedBox(height: 12),
                      
                      ..._classificationResult!['all_probabilities'].entries.map((entry) {
                        double probability = entry.value * 100;
                        return Padding(
                          padding: EdgeInsets.symmetric(vertical: 4),
                          child: Row(
                            children: [
                              Icon(
                                _getLeafIcon(entry.key),
                                size: 20,
                                color: Colors.grey[600],
                              ),
                              SizedBox(width: 8),
                              Expanded(
                                flex: 2,
                                child: Text(
                                  _getLeafDisplayName(entry.key),
                                  style: TextStyle(fontWeight: FontWeight.w500),
                                ),
                              ),
                              Expanded(
                                flex: 3,
                                child: LinearProgressIndicator(
                                  value: probability / 100,
                                  backgroundColor: Colors.grey[300],
                                  valueColor: AlwaysStoppedAnimation<Color>(
                                    _getConfidenceColor(entry.value),
                                  ),
                                ),
                              ),
                              SizedBox(width: 8),
                              Text(
                                '${probability.toStringAsFixed(1)}%',
                                style: TextStyle(
                                  fontWeight: FontWeight.w500,
                                  color: Colors.grey[700],
                                ),
                              ),
                            ],
                          ),
                        );
                      }).toList(),
                    ],
                  ),
                ),
              ),
            
            // Desteklenen Sınıflar Bilgisi
            Card(
              elevation: 2,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              child: Padding(
                padding: EdgeInsets.all(16.0),
                child: Column(
                  children: [
                    Icon(Icons.info_outline, color: Colors.blue[600]),
                    SizedBox(height: 8),
                    Text(
                      'Desteklenen Yaprak Türleri',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    SizedBox(height: 12),
                    Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: supportedClasses.map((className) {
                        return Chip(
                          avatar: Icon(
                            _getLeafIcon(className),
                            size: 18,
                            color: Colors.green[700],
                          ),
                          label: Text(
                            _getLeafDisplayName(className),
                            style: TextStyle(fontSize: 12),
                          ),
                          backgroundColor: Colors.green[50],
                        );
                      }).toList(),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}