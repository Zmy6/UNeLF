# UNeLF: UnconsUNeLF2-9ed Neural Light Field for Self-Supervised Angular Super-Resolution
Compared to supervised learning methods, selfsupervised learning methods address the domain gap problem between light field (LF) datasets collected under varying acquisition conditions, which typically leads to decreased performance when differences exist in the distribution between the training and test sets. However, current self-supervised light field angular superresolution (LFASR) techniques primarily focus on exploiting discrete spatial-angular features while neglecting continuous LF information. In contrast to previous work, we propose a selfsupervised unconstrained neural light field (UNeLF) to continuously represent LF for LFASR. Specifically, any LF can be described as the camera pose for each sub-aperture image (SAI) and the two-plane that captures these SAIs. To describe the former, we introduce a SAIs-dependent pose optimization method to solve the issue that arises from the narrow baseline of most LF data, which hinders robust camera pose estimation. This mechanism reduces the number of trainable camera parameters from a quadratic to a constant scale, thereby alleviating the complexity of joint optimization. For the latter, we propose a novel adaptive two-plane parameterization strategy to determine the two-plane that captures these SAIs, facilitating refocusing. Finally, we jointly optimize the camera parameters, near-far planes and neural light field, efficiently mapping each adaptive two-plane parameterized ray to its correspondence color in a continuous manner. Comprehensive experiments demonstrate that UNeLF achieves faster training and inference with fewer computational resources while exhibiting superior performance on both synthetic and real-world datasets.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

/```
Give examples
/```

### Installing

A step by step series of examples that tell you how to get a development env running

Say what the step will be

/```
Give the example
/```

And repeat

/```
until finished
/```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

/```
Give an example
/```

### And coding style tests

Explain what these tests test and why

/```
Give an example
/```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

