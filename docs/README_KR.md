# 한국어 문서 | [中文](./README_CN.md) | [English](../README.md)
범용 자율주행 시스템 프레임워크
![license](https://img.shields.io/badge/license-MIT-blue.svg)
![language](https://img.shields.io/badge/language-한국어-blue.svg)

## 목차
- [시스템 개요](#시스템-개요)
- [시스템 아키텍처](#시스템-아키텍처)
- [기능 모듈](#기능-모듈)
  - [제어 시스템](#제어-시스템)
  - [계획 시스템](#계획-시스템)
  - [인식 시스템](#인식-시스템)
  - [센싱 시스템](#센싱-시스템)
  - [입출력 인터페이스](#입출력-인터페이스)
- [빠른 시작](#빠른-시작)
- [시스템 확장](#시스템-확장)
- [성능 테스트](#성능-테스트)
- [기여 가이드](#기여-가이드)
- [감사의 글](#감사의-글)
- [라이선스](#라이선스)

## 시스템 개요
이 프레임워크는 다양한 플랫폼에서 사용할 수 있는 범용 자율주행 시스템으로, 유연한 구성과 기능 확장을 지원합니다. 본 시스템은 다양한 하드웨어 시설과 응용 시나리오에 적용할 수 있는 표준화된 자율주행 개발 플랫폼을 제공하는 것을 목표로 합니다.

## 시스템 아키텍처
pass

## 기능 모듈

### 제어 시스템
#### 특징
- 다중 컨트롤러 지원
- 실시간성 보장
- 제어 정밀도 최적화
#### 구현 방식
pass

### 계획 시스템
#### 특징
- 다중 시나리오 적응
- 동적 경로 계획
- 실시간 의사결정 최적화
#### 구현 방식
pass

### 인식 시스템
#### 특징
- 다중 센서 융합
- 객체 감지 및 추적
- 장면 이해 능력
#### 구현 방식
pass

### 센싱 시스템
#### 특징
- 센서 모듈화 설계
- 데이터 동기화 메커니즘
- 오류 감지 및 복구
#### 구현 방식
pass

### 입출력 인터페이스
#### 특징
- 일회성 구성으로 영구 사용
- 표준화된 인터페이스 정의
- 크로스 플랫폼 호환성
#### 구현 방식
pass

## 빠른 시작
### Docker Environment Setup

Run the following command to enter the docker environment:
```bash
./docker/run_container.sh
```
This will pull and run our pre-configured docker image from Docker Hub.

### Development Setup

#### Windows Users
1. Open WSL terminal
2. Run `code .` to open VS Code
3. Connect to docker container within VS Code

## Docker Image

Our official docker image is available on Docker Hub:
```bash
docker push leolixingyou/ros1_for_self-driving:carla_0.9.13
```

## 시스템 확장
pass

## 성능 테스트
pass

## 기여 가이드
프로젝트에 기여하고 싶으시다면 다음 단계를 따라주세요:
1. 저장소를 Fork 하기
2. 기능 브랜치 생성 (`git checkout -b feature/AmazingFeature`)
3. 변경사항 커밋 (`git commit -m 'Add some AmazingFeature'`)
4. 브랜치에 Push (`git push origin feature/AmazingFeature`)
5. Pull Request 열기

## 감사의 글
- 이 프로젝트에 기여한 모든 개발자분들께 감사드립니다
- 다음 기관/개인분들께 특별히 감사드립니다:
  - [추가 예정]
  - [추가 예정]

## 라이선스
이 프로젝트는 MIT 라이선스를 따릅니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

