from enum import IntEnum

class Consciousness(IntEnum):
    ALERT_KEENLY_RESPONSIVE = 0
    AROUSES_TO_MINOR_STIMULATION = 1
    REQUIRES_REPEATED_STIMULATION_TO_AROUSE = 2
    MOVEMENTS_TO_PAIN = 2
    POSTURES_OR_UNRESPONSIVE = 3

class MonthAndAgeQuestions(IntEnum):
    BOTH_QUESTIONS_RIGHT = 0
    ONE_QUESTION_RIGHT = 1
    ZERO_QUESTIONS_RIGHT = 2
    DYSARTHRIC_INTUBATED_TRAUMA_OR_LANGUAGE_BARRIER = 1
    APHASIC = 2

class BlinkEyesAndSqueezeHands(IntEnum):
    PERFORMS_BOTH_TASKS = 0
    PERFORMS_ONE_TASK = 1
    PERFORMS_ZERO_TASKS = 2

class HorizontalExtraocularMovements(IntEnum):
    NORMAL = 0
    PARTIAL_GAZE_PALSY_CAN_BE_OVERCOME = 1
    PARTIAL_GAZE_PALSY_CORRECTS_WITH_OCULOCEPHALIC_REFLEX = 1
    FORCED_GAZE_PALSY_CANNOT_BE_OVERCOME = 2

class VisualFields(IntEnum):
    NO_VISUAL_LOSS = 0
    PARTIAL_HEMIANOPIA = 1
    COMPLETE_HEMIANOPIA = 2
    PATIENT_IS_BILATERALLY_BLIND = 3
    BILATERAL_HEMIANOPIA = 3

class FacialPalsy(IntEnum):
    NORMAL_SYMMETRY = 0
    MINOR_PARALYSIS = 1
    PARTIAL_PARALYSIS = 2
    UNILATERAL_COMPLETE_PARALYSIS = 3
    BILATERAL_COMPLETE_PARALYSIS = 4

class ArmMotorDrift(IntEnum):
    NO_DRIFT_FOR_10_SECONDS = 0
    DRIFTS_BUT_DOES_NOT_HIT_BED = 1
    DRIFTS_HITS_BED = 2
    SOME_EFFORT_AGAINST_GRAVITY = 2
    NO_EFFORT_AGAINST_GRAVITY = 3
    NO_MOVEMENT = 4
    AMPUTATION_JOINT_FUSION = 0

class LegMotorDrift(IntEnum):
    NO_DRIFT_FOR_5_SECONDS = 0
    DRIFTS_BUT_DOES_NOT_HIT_BED = 1
    DRIFTS_HITS_BED = 2
    SOME_EFFORT_AGAINST_GRAVITY = 2
    NO_EFFORT_AGAINST_GRAVITY = 3
    NO_MOVEMENT = 4
    AMPUTATION_JOINT_FUSION = 0

class LimbAtaxia(IntEnum):
    NO_ATAXIA = 0
    ATAXIA_IN_1_LIMB = 1
    ATAXIA_IN_2_LIMBS = 2
    DOES_NOT_UNDERSTAND = 0
    PARALYZED = 0
    AMPUTATION_JOINT_FUSION = 0

class Sensation(IntEnum):
    NORMAL_NO_SENSORY_LOSS = 0
    MILD_MODERATE_LOSS_LESS_SHARP_MORE_DULL = 1
    MILD_MODERATE_LOSS_CAN_SENSE_BEING_TOUCHED = 1
    COMPLETE_LOSS_CANNOT_SENSE_BEING_TOUCHED_AT_ALL = 2
    NO_RESPONSE_AND_QUADRIPLEGIC = 2
    COMA_UNRESPONSIVE = 2

class LanguageAphasia(IntEnum):
    # Describe the scene; name the items; read the sentences
    NORMAL_NO_APHASIA = 0
    MILD_MODERATE_APHASIA_SOME_OBVIOUS_CHAGNES_WITHOUT_SIGNIFICANT_LIMITATION = 1
    SEVERE_APHASIA_FRAGMENTARY_EXPRESSION_INFERENCE_NEEDED_CANNOT_IDENTIFY_MATERIALS = 2
    MUTE_GLOBAL_APHASIA_NO_USABLE_SPEECH_AUDITORY_COMPREHENSION = 3
    COMA_UNRESPONSIVE = 3

class Dysarthria(IntEnum):
    # Read the words
    NORMAL = 0
    MILD_MODERATE_DYSARTHRIA_SLURRING_BUT_CAN_BE_UNDERSTOOD = 1
    SEVERE_DYSARTHRIA_UNINTELLIGIBLE_SLURRING_OR_OUT_OF_PROPORTION_TO_DYSPHAGIA = 2
    MUTE_ANARTHRIC = 2
    INTUBATED_UNABLE_TO_TEST = 0

class ExtinctionInattention(IntEnum):
    NO_ABNORMALITY = 0
    VISUAL_TACTILE_AUDITORY_SPATIAL_PERSONAL_INATTENTION = 1
    EXTINCTION_TO_BILATERAL_SIMULTANEOUS_STIMULATION = 1
    PROFOUND_HEMI_INATTENTION = 2
    EXTINCTION_TO_GREATER_THAN_1_MODALITY = 2

def nihss(consciousness: Consciousness,
          month_and_age_questions: MonthAndAgeQuestions,
          blink_eyes_and_squeeze_hands: BlinkEyesAndSqueezeHands,
          horizontal_extraocular_movements: HorizontalExtraocularMovements,
          visual_fields: VisualFields,
          facial_palsy: FacialPalsy,
          left_arm_motor_drift: ArmMotorDrift,
          right_arm_motor_drift: ArmMotorDrift,
          left_leg_motor_drift: LegMotorDrift,
          right_leg_motor_drift: LegMotorDrift,
          limb_ataxia: LimbAtaxia,
          sensation: Sensation,
          language: LanguageAphasia,
          dysarthria: Dysarthria,
          inattention: ExtinctionInattention) -> int:
    """
    Paramters:
    - month_and_age_questions: The patient's response to being asked for their age and the month of the year.
    - blink_eyes_and_squeeze_hands: The patient's response to being asked to blink their eyes and squeeze the examiner's hand. Can be pantomimed if patient does not understand.
    - horizontal_extraocular_movements: The patient's ability to perform horizontal extraocular movements. (i.e. look left and right with both eyes)
    - visual_fields: The integrity of the patient's visual fields.
    - facial_palsy: The ability of the patient to engage their facial muscles.
    - left_arm_motor_drift: The patient's ability to overcome gravity with their left arm and hold a position without drifting.
    - right_arm_motor_drift: The patient's ability to overcome gravity with their right arm and hold a position without drifting.
    - left_leg_motor_drift: The patient's ability to overcome gravity with their left leg and hold a position without drifting.
    - right_leg_motor_drift: The patient's ability to overcome gravity with their right leg and hold a position without drifting.
    - limb_ataxia: The patient's ability to coordinate their limbs.
    - sensation: The patients's ability to sense touch.
    - language: The patient's ability to understand and construct language or lack thereof (i.e. aphasia).
    - dysarthria: The patient's ability to produce the sounds of language or lack thereof (i.e. dysarthria).
    - inattention: The patient's ability to maintain attention or lack thereof.
    Returns: The NIH Stroke Scale score for this patient.
    """
    
    return sum((consciousness,
                month_and_age_questions,
                blink_eyes_and_squeeze_hands,
                horizontal_extraocular_movements,
                visual_fields,
                facial_palsy,
                left_arm_motor_drift,
                right_arm_motor_drift,
                left_leg_motor_drift,
                right_leg_motor_drift,
                limb_ataxia,
                sensation,
                language,
                dysarthria,
                inattention))