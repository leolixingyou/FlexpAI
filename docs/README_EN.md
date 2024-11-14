# Universal Autonomous Driving System Framework | [中文](../README.md) | [한국어](./README_KR.md)

![license](https://img.shields.io/badge/license-MIT-blue.svg)
![language](https://img.shields.io/badge/language-English-blue.svg)

[中文文档](../README.md) | [한국어 문서](./README_KR.md)

## Table of Contents
- [System Overview](#system-overview)
- [System Architecture](#system-architecture)
- [Functional Modules](#functional-modules)
  - [Control System](#control-system)
  - [Planning System](#planning-system)
  - [Perception System](#perception-system)
  - [Sensing System](#sensing-system)
  - [I/O Interface](#io-interface)
- [Quick Start](#quick-start)
- [System Extension](#system-extension)
- [Performance Testing](#performance-testing)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## System Overview
This is a multi-platform universal autonomous driving system framework that supports flexible configuration and functional extension. The system aims to provide a standardized autonomous driving development platform that can adapt to different hardware facilities and application scenarios.

## System Architecture
pass

## Functional Modules

### Control System
#### Features
- Multiple controller support
- Real-time guarantee
- Control precision optimization
#### Implementation
pass

### Planning System
#### Features
- Multi-scenario adaptation
- Dynamic path planning
- Real-time decision optimization
#### Implementation
pass

### Perception System
#### Features
- Multi-sensor fusion
- Object detection and tracking
- Scene understanding capability
#### Implementation
pass

### Sensing System
#### Features
- Modular sensor design
- Data synchronization mechanism
- Fault detection and recovery
#### Implementation
pass

### I/O Interface
#### Features
- One-time configuration for permanent use
- Standardized interface definition
- Cross-platform compatibility
#### Implementation
pass

## Quick Start
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


## System Extension
pass

## Performance Testing
pass

## Contributing
If you want to contribute to this project, please follow these steps:
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments
- Thanks to all developers who contributed to this project
- Special thanks to the following organizations/individuals:
  - [To be added]
  - [To be added]

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.