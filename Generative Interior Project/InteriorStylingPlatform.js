import React, { useState, useCallback } from 'react';
import { Upload, Image as LucideImage, Palette, ShoppingBag, RefreshCcw, X } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';

const InteriorStylingPlatform = () => {
  const [activeStep, setActiveStep] = useState(1);
  const [selectedStyle, setSelectedStyle] = useState('');
  const [uploadedImage, setUploadedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  
  const styles = [
    { 
      id: 'modern', 
      name: '모던', 
      description: '깔끔한 라인과 미니멀한 장식',
      example: '/api/placeholder/200/150'
    },
    { 
      id: 'scandinavian', 
      name: '스칸디나비안', 
      description: '밝은 색상과 자연 소재',
      example: '/api/placeholder/200/150'
    },
    { 
      id: 'industrial', 
      name: '인더스트리얼', 
      description: '원자재와 노출된 요소들',
      example: '/api/placeholder/200/150'
    },
  ];

  const recommendedFurniture = [
    { 
      id: 1, 
      name: '미니멀 소파', 
      price: '₩1,200,000', 
      store: '네이버쇼핑',
      image: '/api/placeholder/100/100'
    },
    { 
      id: 2, 
      name: '커피 테이블', 
      price: '₩450,000', 
      store: '쿠팡',
      image: '/api/placeholder/100/100'
    },
    { 
      id: 3, 
      name: '플로어 램프', 
      price: '₩180,000', 
      store: '오늘의집',
      image: '/api/placeholder/100/100'
    },
  ];

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    const file = e.dataTransfer?.files[0] || e.target.files[0];
    
    if (file) {
      if (!file.type.startsWith('image/')) {
        setError('이미지 파일만 업로드 가능합니다.');
        return;
      }

      const reader = new FileReader();
      reader.onload = () => {
        setPreviewUrl(reader.result);
        setUploadedImage(file);
        setError('');
        setActiveStep(2);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
  }, []);

  const removeImage = () => {
    setPreviewUrl('');
    setUploadedImage(null);
    setActiveStep(1);
  };

  const generateStyle = async () => {
    if (!uploadedImage || !selectedStyle) {
      setError('이미지와 스타일을 모두 선택해주세요.');
      return;
    }

    setLoading(true);
    // AI 스타일링 생성 로직
    await new Promise(resolve => setTimeout(resolve, 2000));
    setLoading(false);
    setActiveStep(3);
  };

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">AI 인테리어 스타일링</h1>
        
        {/* Progress Steps */}
        <div className="flex justify-between mb-12">
          {[
            { step: 1, label: '이미지 업로드' },
            { step: 2, label: '스타일 선택' },
            { step: 3, label: 'AI 스타일링' },
            { step: 4, label: '가구 추천' }
          ].map(({ step, label }) => (
            <div key={step} className="flex flex-col items-center">
              <div className="flex items-center">
                <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                  activeStep >= step ? 'bg-blue-600 text-white' : 'bg-gray-200'
                }`}>
                  {step}
                </div>
                {step < 4 && (
                  <div className={`w-24 h-1 ${
                    activeStep > step ? 'bg-blue-600' : 'bg-gray-200'
                  }`} />
                )}
              </div>
              <span className="text-sm mt-2">{label}</span>
            </div>
          ))}
        </div>

        {error && (
          <Alert className="mb-6 bg-red-50 border-red-200">
            <AlertDescription className="text-red-800">{error}</AlertDescription>
          </Alert>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Left Column */}
          <div className="space-y-6">
            {/* Upload Section */}
            <Card>
              <CardHeader>
                <CardTitle>공간 이미지 업로드</CardTitle>
              </CardHeader>
              <CardContent>
                <div
                  className="relative border-2 border-dashed border-gray-300 rounded-lg p-12 text-center"
                  onDrop={handleDrop}
                  onDragOver={handleDragOver}
                >
                  {previewUrl ? (
                    <div className="relative">
                      <img 
                        src={previewUrl} 
                        alt="업로드된 이미지" 
                        className="mx-auto max-h-64 rounded-lg"
                      />
                      <button
                        onClick={removeImage}
                        className="absolute top-2 right-2 p-1 bg-white rounded-full shadow-lg"
                      >
                        <X className="h-4 w-4 text-gray-600" />
                      </button>
                    </div>
                  ) : (
                    <>
                      <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                      <p className="text-gray-600 mb-2">이미지를 드래그하거나 클릭하여 업로드하세요</p>
                      <p className="text-sm text-gray-500">권장 크기: 1920x1080px</p>
                    </>
                  )}
                  <input
                    type="file"
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                    onChange={handleDrop}
                    accept="image/*"
                  />
                </div>
              </CardContent>
            </Card>

            {/* Style Selection */}
            <Card>
              <CardHeader>
                <CardTitle>인테리어 스타일 선택</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid gap-4">
                  {styles.map((style) => (
                    <div
                      key={style.id}
                      className={`p-4 border rounded-lg cursor-pointer transition-all ${
                        selectedStyle === style.id 
                          ? 'border-blue-600 bg-blue-50 shadow-md' 
                          : 'border-gray-200 hover:border-blue-300'
                      }`}
                      onClick={() => {
                        setSelectedStyle(style.id);
                        if (uploadedImage) setActiveStep(3);
                      }}
                    >
                      <div className="flex items-center gap-4">
                        <img
                          src={style.example}
                          alt={style.name}
                          className="w-24 h-16 object-cover rounded"
                        />
                        <div>
                          <h3 className="font-medium">{style.name}</h3>
                          <p className="text-sm text-gray-600">{style.description}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Right Column */}
          <div className="space-y-6">
            {/* Generated Result */}
            <Card>
              <CardHeader>
                <CardTitle>AI 스타일링 결과</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="aspect-video bg-gray-100 rounded-lg flex items-center justify-center">
                  {loading ? (
                    <div className="flex flex-col items-center gap-2">
                      <RefreshCcw className="h-8 w-8 text-blue-600 animate-spin" />
                      <p className="text-sm text-gray-600">AI가 스타일링을 생성중입니다...</p>
                    </div>
                  ) : (
                    <LucideImage className="h-12 w-12 text-gray-400" />
                  )}
                </div>
                <button 
                  className="mt-4 w-full flex items-center justify-center gap-2 bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-300"
                  onClick={generateStyle}
                  disabled={loading || !uploadedImage || !selectedStyle}
                >
                  <RefreshCcw className="h-4 w-4" />
                  스타일링 생성하기
                </button>
              </CardContent>
            </Card>

            {/* Recommended Furniture */}
            <Card>
              <CardHeader>
                <CardTitle>추천 가구</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {recommendedFurniture.map((item) => (
                    <div key={item.id} className="flex items-center gap-4 p-4 border rounded-lg hover:border-blue-300 transition-colors">
                      <img
                        src={item.image}
                        alt={item.name}
                        className="w-16 h-16 object-cover rounded"
                      />
                      <div className="flex-1">
                        <h3 className="font-medium">{item.name}</h3>
                        <p className="text-sm text-gray-600">{item.price}</p>
                      </div>
                      <button className="flex items-center gap-2 text-blue-600 hover:text-blue-700">
                        <ShoppingBag className="h-4 w-4" />
                        <span className="text-sm">{item.store}에서 구매</span>
                      </button>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default InteriorStylingPlatform;
