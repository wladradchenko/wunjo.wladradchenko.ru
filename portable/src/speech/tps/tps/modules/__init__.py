from speech.tps.tps.modules.processor import Processor

from speech.tps.tps.modules.custom.replacer import Replacer, BlindReplacer
from speech.tps.tps.modules.custom.auxiliary import Lower, Cleaner, Number

from speech.tps.tps.modules.emphasizer.rule_based.independent import Emphasizer
from speech.tps.tps.modules.emphasizer.rule_based.russian import RuEmphasizer

from speech.tps.tps.modules.omographs.rule_based.independent import Omograph
from speech.tps.tps.modules.omographs.rule_based.russian import RuOmograph

from speech.tps.tps.modules.phonetizer.rule_based.independent import Phonetizer

from speech.tps.tps.modules.polyphonic.rule_based.chinese import ZhPolyphonic