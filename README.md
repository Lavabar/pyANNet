# Accuracy scores
- epochs=3, minibatch=128, optimizer=Adam

| framework+architecture | activations                   | test accuracy |
|------------------------|-------------------------------|---------------|
| ours fcnn 128,64       | all softplus                  | 0.9523        |
| ours fcnn 128,64       | all relus                     | 0.9476        |
| ours cnn lenet         | all softplus                  | 0.8815        |
| ours cnn lenet         | all relus                     | 0.9385        |
|                        |                               |               |
| keras fcnn 128,64      | all softplus                  | 0.9599        |
| keras fcnn 128,64      | all relus                     | 0.9722        |
| keras cnn lenet        | all softplus                  | 0,9750        |
| keras cnn lenet        | all relus                     | 0.9846        |
| keras cnn lenet        | conv relus and dense softplus | 0.9794        |