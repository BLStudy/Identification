## Unique identification of 50,000+ virtual reality users from their head and hand motion data

With the recent explosive growth of interest and investment in virtual reality (VR) and the so-called "metaverse," public attention has rightly shifted toward the unique security and privacy threats that these platforms may pose. While it has long been known that people reveal information about themselves via their motion, the extent to which this makes an individual globally identifiable within virtual reality has not yet been widely understood. In this study, we show that a large number of real VR users (N=55,541) can be uniquely and reliably identified across multiple sessions using just their head and hand motion relative to virtual objects. After training a classification model on 5 minutes of data per person, a user can be uniquely identified amongst the entire pool of 50,000+ with 94.33% accuracy from 100 seconds of motion, and with 73.20% accuracy from just 10 seconds of motion. This work is the first to truly demonstrate the extent to which biomechanics may serve as a unique identifier in VR, on par with widely used biometrics such as facial or fingerprint recognition.

_Repository anonymized for blinded conference review_

**Contents:**
- Featurization: `featurzation/`
  - Run `node parse.js`
- Normalization: `00-`
  - Run `py 00-normalize.py`
- Training: `01-` through `05-`
  - To train layers 1 and 2, run `py 01-train_layer_1.py` and `py 02-train_layer_2.py`
  - To form clusters for layer 3, run `py 03-test_confusion.py` and `py 04-group.py`
  - To train layer 3, run `05-train_layer_3.py`
- Testing: `06-`
  - Run `py 06-test_final.py`
- Explanation: `explain.py` and `explain2.py`
  - Run `py explain.py` and `py explain2.py`
  
**Data:** For privacy and logistical reasons, we have not published the full training dataset (3.96 TB). Researchers may contact the authors if they wish to obtain a copy for replication purposes.
