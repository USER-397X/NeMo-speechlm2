# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Text normalization for WER/CER calculation.
100% compliant with HuggingFace Open ASR Leaderboard protocol.

Adapted from OpenAI Whisper normalizer with extensions for multilingual support.
"""

import re
import unicodedata
from fractions import Fraction
from typing import Iterator, List, Match, Optional, Union

try:
    import regex
except ImportError:
    regex = None


def remove_chat_template_roles(s: str) -> str:
    """
    Remove chat template role markers from text for accurate WER/CER calculation.

    This function removes role markers that are part of chat templates (e.g., ChatML format)
    but should NOT be included in ASR metric calculations. It handles:
    - Special tokens: <|im_start|>, <|im_end|>, <|endoftext|>, </think>
    - Role markers at string start: "assistant", "user", "system" (case-insensitive)
    - Repeated role markers: "assistantassistant assistant" → removed entirely
    - Role markers with or without spaces: "assistantI" or "assistant I" → "I"

    Important: Only removes role markers at the START of the string to avoid incorrectly
    removing legitimate words like "The assistant helped me" in actual transcriptions.

    Args:
        s: Input text potentially containing chat template markers

    Returns:
        Cleaned text with chat template markers removed

    Examples:
        >>> remove_chat_template_roles("<|im_start|>assistant Hello world")
        "Hello world"
        >>> remove_chat_template_roles("assistantassistant assistant I think")
        "I think"
        >>> remove_chat_template_roles("</think>assistant Hello")
        "Hello"
        >>> remove_chat_template_roles("The assistant helped me")  # NOT removed
        "The assistant helped me"
    """
    if not s:
        return s

    # Remove special tokens (angle bracket format and other thinking tokens)
    # Handles: <|im_start|>, <|im_end|>, <|endoftext|>, </think>, etc.
    s = re.sub(r"<\|[^|]+\|>", "", s)
    s = re.sub(r"</think>", "", s, flags=re.IGNORECASE)

    # Remove role markers at the start of the string
    # Pattern explanation:
    # ^             - Start of string
    # (             - Start capture group
    #   \s*         - Optional whitespace
    #   (assistant|user|system)  - Role marker (case-insensitive)
    # )+            - One or more repetitions (handles "assistantassistant assistant")
    # \s*           - Optional trailing whitespace
    #
    # This pattern handles:
    # - "assistant hello" → "hello"
    # - "assistantassistant I" → "I"
    # - "assistant assistant hello" → "hello"
    # - "  assistant  hello" → "hello"
    s = re.sub(r"^(\s*(assistant|user|system))+\s*", "", s, flags=re.IGNORECASE)

    # Clean up any extra whitespace
    s = s.strip()

    return s


# Non-ASCII letters that are not separated by "NFKD" normalization
ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}


# British-American spelling mappings
# Reference: https://www.tysto.com/uk-us-spelling-list.html
english_spelling_normalizer = {
    "accessorise": "accessorize",
    "accessorised": "accessorized",
    "accessorises": "accessorizes",
    "accessorising": "accessorizing",
    "acclimatisation": "acclimatization",
    "acclimatise": "acclimatize",
    "acclimatised": "acclimatized",
    "acclimatises": "acclimatizes",
    "acclimatising": "acclimatizing",
    "accoutrements": "accouterments",
    "aeon": "eon",
    "aeons": "eons",
    "aerogramme": "aerogram",
    "aerogrammes": "aerograms",
    "aeroplane": "airplane",
    "aeroplanes": "airplanes",
    "aesthete": "esthete",
    "aesthetes": "esthetes",
    "aesthetic": "esthetic",
    "aesthetically": "esthetically",
    "aesthetics": "esthetics",
    "aetiology": "etiology",
    "ageing": "aging",
    "aggrandisement": "aggrandizement",
    "agonise": "agonize",
    "agonised": "agonized",
    "agonises": "agonizes",
    "agonising": "agonizing",
    "agonisingly": "agonizingly",
    "almanack": "almanac",
    "almanacks": "almanacs",
    "aluminium": "aluminum",
    "amortisable": "amortizable",
    "amortisation": "amortization",
    "amortisations": "amortizations",
    "amortise": "amortize",
    "amortised": "amortized",
    "amortises": "amortizes",
    "amortising": "amortizing",
    "amphitheatre": "amphitheater",
    "amphitheatres": "amphitheaters",
    "anaemia": "anemia",
    "anaemic": "anemic",
    "anaesthesia": "anesthesia",
    "anaesthetic": "anesthetic",
    "anaesthetics": "anesthetics",
    "anaesthetise": "anesthetize",
    "anaesthetised": "anesthetized",
    "anaesthetises": "anesthetizes",
    "anaesthetising": "anesthetizing",
    "anaesthetist": "anesthetist",
    "anaesthetists": "anesthetists",
    "anaesthetize": "anesthetize",
    "anaesthetized": "anesthetized",
    "anaesthetizes": "anesthetizes",
    "anaesthetizing": "anesthetizing",
    "analogue": "analog",
    "analogues": "analogs",
    "analyse": "analyze",
    "analysed": "analyzed",
    "analyses": "analyzes",
    "analysing": "analyzing",
    "anglicise": "anglicize",
    "anglicised": "anglicized",
    "anglicises": "anglicizes",
    "anglicising": "anglicizing",
    "annualised": "annualized",
    "antagonise": "antagonize",
    "antagonised": "antagonized",
    "antagonises": "antagonizes",
    "antagonising": "antagonizing",
    "apologise": "apologize",
    "apologised": "apologized",
    "apologises": "apologizes",
    "apologising": "apologizing",
    "appal": "appall",
    "appals": "appalls",
    "appetiser": "appetizer",
    "appetisers": "appetizers",
    "appetising": "appetizing",
    "appetisingly": "appetizingly",
    "arbour": "arbor",
    "arbours": "arbors",
    "archaeologically": "archeologically",
    "archaeologist": "archeologist",
    "archaeologists": "archeologists",
    "archaeology": "archeology",
    "archeological": "archaeological",
    "ardour": "ardor",
    "armour": "armor",
    "armoured": "armored",
    "armourer": "armorer",
    "armourers": "armorers",
    "armouries": "armories",
    "armoury": "armory",
    "artefact": "artifact",
    "artefacts": "artifacts",
    "authorise": "authorize",
    "authorised": "authorized",
    "authorises": "authorizes",
    "authorising": "authorizing",
    "axe": "ax",
    "backpedalled": "backpedaled",
    "backpedalling": "backpedaling",
    "bannister": "banister",
    "bannisters": "banisters",
    "baptise": "baptize",
    "baptised": "baptized",
    "baptises": "baptizes",
    "baptising": "baptizing",
    "bastardise": "bastardize",
    "bastardised": "bastardized",
    "bastardises": "bastardizes",
    "bastardising": "bastardizing",
    "battleax": "battleaxe",
    "baulk": "balk",
    "baulked": "balked",
    "baulking": "balking",
    "baulks": "balks",
    "bedevilled": "bedeviled",
    "bedevilling": "bedeviling",
    "behaviour": "behavior",
    "behavioural": "behavioral",
    "behaviourism": "behaviorism",
    "behaviourist": "behaviorist",
    "behaviourists": "behaviorists",
    "behaviours": "behaviors",
    "behove": "behoove",
    "behoved": "behooved",
    "behoves": "behooves",
    "bejewelled": "bejeweled",
    "belabour": "belabor",
    "belaboured": "belabored",
    "belabouring": "belaboring",
    "belabours": "belabors",
    "bevelled": "beveled",
    "bevvies": "bevies",
    "bevvy": "bevy",
    "biassed": "biased",
    "biassing": "biasing",
    "bingeing": "binging",
    "bougainvillaea": "bougainvillea",
    "bougainvillaeas": "bougainvilleas",
    "bowdlerise": "bowdlerize",
    "bowdlerised": "bowdlerized",
    "bowdlerises": "bowdlerizes",
    "bowdlerising": "bowdlerizing",
    "breathalyse": "breathalyze",
    "breathalysed": "breathalyzed",
    "breathalyser": "breathalyzer",
    "breathalysers": "breathalyzers",
    "breathalyses": "breathalyzes",
    "breathalysing": "breathalyzing",
    "brutalise": "brutalize",
    "brutalised": "brutalized",
    "brutalises": "brutalizes",
    "brutalising": "brutalizing",
    "busses": "buses",
    "bussing": "busing",
    "caesarean": "cesarean",
    "caesareans": "cesareans",
    "calibre": "caliber",
    "calibres": "calibers",
    "calliper": "caliper",
    "callipers": "calipers",
    "callisthenics": "calisthenics",
    "canalise": "canalize",
    "canalised": "canalized",
    "canalises": "canalizes",
    "canalising": "canalizing",
    "cancelation": "cancellation",
    "cancelations": "cancellations",
    "cancelled": "canceled",
    "cancelling": "canceling",
    "candour": "candor",
    "cannibalise": "cannibalize",
    "cannibalised": "cannibalized",
    "cannibalises": "cannibalizes",
    "cannibalising": "cannibalizing",
    "canonise": "canonize",
    "canonised": "canonized",
    "canonises": "canonizes",
    "canonising": "canonizing",
    "capitalise": "capitalize",
    "capitalised": "capitalized",
    "capitalises": "capitalizes",
    "capitalising": "capitalizing",
    "caramelise": "caramelize",
    "caramelised": "caramelized",
    "caramelises": "caramelizes",
    "caramelising": "caramelizing",
    "carbonise": "carbonize",
    "carbonised": "carbonized",
    "carbonises": "carbonizes",
    "carbonising": "carbonizing",
    "carolled": "caroled",
    "carolling": "caroling",
    "catalogue": "catalog",
    "catalogued": "cataloged",
    "catalogues": "catalogs",
    "cataloguing": "cataloging",
    "catalyse": "catalyze",
    "catalysed": "catalyzed",
    "catalyses": "catalyzes",
    "catalysing": "catalyzing",
    "categorise": "categorize",
    "categorised": "categorized",
    "categorises": "categorizes",
    "categorising": "categorizing",
    "cauterise": "cauterize",
    "cauterised": "cauterized",
    "cauterises": "cauterizes",
    "cauterising": "cauterizing",
    "cavilled": "caviled",
    "cavilling": "caviling",
    "centigramme": "centigram",
    "centigrammes": "centigrams",
    "centilitre": "centiliter",
    "centilitres": "centiliters",
    "centimetre": "centimeter",
    "centimetres": "centimeters",
    "centralise": "centralize",
    "centralised": "centralized",
    "centralises": "centralizes",
    "centralising": "centralizing",
    "centre": "center",
    "centred": "centered",
    "centrefold": "centerfold",
    "centrefolds": "centerfolds",
    "centrepiece": "centerpiece",
    "centrepieces": "centerpieces",
    "centres": "centers",
    "channelled": "channeled",
    "channelling": "channeling",
    "characterise": "characterize",
    "characterised": "characterized",
    "characterises": "characterizes",
    "characterising": "characterizing",
    "cheque": "check",
    "chequebook": "checkbook",
    "chequebooks": "checkbooks",
    "chequered": "checkered",
    "cheques": "checks",
    "chilli": "chili",
    "chimaera": "chimera",
    "chimaeras": "chimeras",
    "chiselled": "chiseled",
    "chiselling": "chiseling",
    "circularise": "circularize",
    "circularised": "circularized",
    "circularises": "circularizes",
    "circularising": "circularizing",
    "civilise": "civilize",
    "civilised": "civilized",
    "civilises": "civilizes",
    "civilising": "civilizing",
    "clamour": "clamor",
    "clamoured": "clamored",
    "clamouring": "clamoring",
    "clamours": "clamors",
    "clangour": "clangor",
    "clarinettist": "clarinetist",
    "clarinettists": "clarinetists",
    "collectivise": "collectivize",
    "collectivised": "collectivized",
    "collectivises": "collectivizes",
    "collectivising": "collectivizing",
    "colonisation": "colonization",
    "colonise": "colonize",
    "colonised": "colonized",
    "coloniser": "colonizer",
    "colonisers": "colonizers",
    "colonises": "colonizes",
    "colonising": "colonizing",
    "colour": "color",
    "colourant": "colorant",
    "colourants": "colorants",
    "coloured": "colored",
    "coloureds": "coloreds",
    "colourful": "colorful",
    "colourfully": "colorfully",
    "colouring": "coloring",
    "colourize": "colorize",
    "colourized": "colorized",
    "colourizes": "colorizes",
    "colourizing": "colorizing",
    "colourless": "colorless",
    "colours": "colors",
    "commercialise": "commercialize",
    "commercialised": "commercialized",
    "commercialises": "commercializes",
    "commercialising": "commercializing",
    "compartmentalise": "compartmentalize",
    "compartmentalised": "compartmentalized",
    "compartmentalises": "compartmentalizes",
    "compartmentalising": "compartmentalizing",
    "computerise": "computerize",
    "computerised": "computerized",
    "computerises": "computerizes",
    "computerising": "computerizing",
    "conceptualise": "conceptualize",
    "conceptualised": "conceptualized",
    "conceptualises": "conceptualizes",
    "conceptualising": "conceptualizing",
    "connexion": "connection",
    "connexions": "connections",
    "contextualise": "contextualize",
    "contextualised": "contextualized",
    "contextualises": "contextualizes",
    "contextualising": "contextualizing",
    "cosier": "cozier",
    "cosies": "cozies",
    "cosiest": "coziest",
    "cosily": "cozily",
    "cosiness": "coziness",
    "cosy": "cozy",
    "councillor": "councilor",
    "councillors": "councilors",
    "counselled": "counseled",
    "counselling": "counseling",
    "counsellor": "counselor",
    "counsellors": "counselors",
    "crenelated": "crenellated",
    "criminalise": "criminalize",
    "criminalised": "criminalized",
    "criminalises": "criminalizes",
    "criminalising": "criminalizing",
    "criticise": "criticize",
    "criticised": "criticized",
    "criticises": "criticizes",
    "criticising": "criticizing",
    "crueller": "crueler",
    "cruellest": "cruelest",
    "crystallisation": "crystallization",
    "crystallise": "crystallize",
    "crystallised": "crystallized",
    "crystallises": "crystallizes",
    "crystallising": "crystallizing",
    "cudgelled": "cudgeled",
    "cudgelling": "cudgeling",
    "customise": "customize",
    "customised": "customized",
    "customises": "customizes",
    "customising": "customizing",
    "cypher": "cipher",
    "cyphers": "ciphers",
    "decentralisation": "decentralization",
    "decentralise": "decentralize",
    "decentralised": "decentralized",
    "decentralises": "decentralizes",
    "decentralising": "decentralizing",
    "decriminalisation": "decriminalization",
    "decriminalise": "decriminalize",
    "decriminalised": "decriminalized",
    "decriminalises": "decriminalizes",
    "decriminalising": "decriminalizing",
    "defence": "defense",
    "defenceless": "defenseless",
    "defences": "defenses",
    "dehumanisation": "dehumanization",
    "dehumanise": "dehumanize",
    "dehumanised": "dehumanized",
    "dehumanises": "dehumanizes",
    "dehumanising": "dehumanizing",
    "demeanour": "demeanor",
    "demilitarisation": "demilitarization",
    "demilitarise": "demilitarize",
    "demilitarised": "demilitarized",
    "demilitarises": "demilitarizes",
    "demilitarising": "demilitarizing",
    "demobilisation": "demobilization",
    "demobilise": "demobilize",
    "demobilised": "demobilized",
    "demobilises": "demobilizes",
    "demobilising": "demobilizing",
    "democratisation": "democratization",
    "democratise": "democratize",
    "democratised": "democratized",
    "democratises": "democratizes",
    "democratising": "democratizing",
    "demonise": "demonize",
    "demonised": "demonized",
    "demonises": "demonizes",
    "demonising": "demonizing",
    "demoralisation": "demoralization",
    "demoralise": "demoralize",
    "demoralised": "demoralized",
    "demoralises": "demoralizes",
    "demoralising": "demoralizing",
    "denationalisation": "denationalization",
    "denationalise": "denationalize",
    "denationalised": "denationalized",
    "denationalises": "denationalizes",
    "denationalising": "denationalizing",
    "deodorise": "deodorize",
    "deodorised": "deodorized",
    "deodorises": "deodorizes",
    "deodorising": "deodorizing",
    "depersonalise": "depersonalize",
    "depersonalised": "depersonalized",
    "depersonalises": "depersonalizes",
    "depersonalising": "depersonalizing",
    "deputise": "deputize",
    "deputised": "deputized",
    "deputises": "deputizes",
    "deputising": "deputizing",
    "desensitisation": "desensitization",
    "desensitise": "desensitize",
    "desensitised": "desensitized",
    "desensitises": "desensitizes",
    "desensitising": "desensitizing",
    "destabilisation": "destabilization",
    "destabilise": "destabilize",
    "destabilised": "destabilized",
    "destabilises": "destabilizes",
    "destabilising": "destabilizing",
    "dialled": "dialed",
    "dialling": "dialing",
    "dialogue": "dialog",
    "dialogues": "dialogs",
    "diarrhoea": "diarrhea",
    "digitise": "digitize",
    "digitised": "digitized",
    "digitises": "digitizes",
    "digitising": "digitizing",
    "disc": "disk",
    "discolour": "discolor",
    "discoloured": "discolored",
    "discolouring": "discoloring",
    "discolours": "discolors",
    "discs": "disks",
    "disembowelled": "disemboweled",
    "disembowelling": "disemboweling",
    "disfavour": "disfavor",
    "dishevelled": "disheveled",
    "dishonour": "dishonor",
    "dishonourable": "dishonorable",
    "dishonourably": "dishonorably",
    "dishonoured": "dishonored",
    "dishonouring": "dishonoring",
    "dishonours": "dishonors",
    "disorganisation": "disorganization",
    "disorganised": "disorganized",
    "distil": "distill",
    "distils": "distills",
    "dramatisation": "dramatization",
    "dramatisations": "dramatizations",
    "dramatise": "dramatize",
    "dramatised": "dramatized",
    "dramatises": "dramatizes",
    "dramatising": "dramatizing",
    "draught": "draft",
    "draughtboard": "draftboard",
    "draughtboards": "draftboards",
    "draughtier": "draftier",
    "draughtiest": "draftiest",
    "draughts": "drafts",
    "draughtsman": "draftsman",
    "draughtsmanship": "draftsmanship",
    "draughtsmen": "draftsmen",
    "draughtswoman": "draftswoman",
    "draughtswomen": "draftswomen",
    "draughty": "drafty",
    "drivelled": "driveled",
    "drivelling": "driveling",
    "duelled": "dueled",
    "duelling": "dueling",
    "economise": "economize",
    "economised": "economized",
    "economises": "economizes",
    "economising": "economizing",
    "editorialise": "editorialize",
    "editorialised": "editorialized",
    "editorialises": "editorializes",
    "editorialising": "editorializing",
    "edoema": "edema",
    "empathise": "empathize",
    "empathised": "empathized",
    "empathises": "empathizes",
    "empathising": "empathizing",
    "emphasise": "emphasize",
    "emphasised": "emphasized",
    "emphasises": "emphasizes",
    "emphasising": "emphasizing",
    "enamelled": "enameled",
    "enamelling": "enameling",
    "enamoured": "enamored",
    "encyclopaedia": "encyclopedia",
    "encyclopaedias": "encyclopedias",
    "encyclopaedic": "encyclopedic",
    "endeavour": "endeavor",
    "endeavoured": "endeavored",
    "endeavouring": "endeavoring",
    "endeavours": "endeavors",
    "energise": "energize",
    "energised": "energized",
    "energises": "energizes",
    "energising": "energizing",
    "enrol": "enroll",
    "enrols": "enrolls",
    "enthral": "enthrall",
    "enthrals": "enthralls",
    "epaulette": "epaulet",
    "epaulettes": "epaulets",
    "epicentre": "epicenter",
    "epicentres": "epicenters",
    "epilogue": "epilog",
    "epilogues": "epilogs",
    "epitomise": "epitomize",
    "epitomised": "epitomized",
    "epitomises": "epitomizes",
    "epitomising": "epitomizing",
    "equalisation": "equalization",
    "equalise": "equalize",
    "equalised": "equalized",
    "equaliser": "equalizer",
    "equalisers": "equalizers",
    "equalises": "equalizes",
    "equalising": "equalizing",
    "eulogise": "eulogize",
    "eulogised": "eulogized",
    "eulogises": "eulogizes",
    "eulogising": "eulogizing",
    "evangelise": "evangelize",
    "evangelised": "evangelized",
    "evangelises": "evangelizes",
    "evangelising": "evangelizing",
    "exorcise": "exorcize",
    "exorcised": "exorcized",
    "exorcises": "exorcizes",
    "exorcising": "exorcizing",
    "extemporisation": "extemporization",
    "extemporise": "extemporize",
    "extemporised": "extemporized",
    "extemporises": "extemporizes",
    "extemporising": "extemporizing",
    "externalisation": "externalization",
    "externalisations": "externalizations",
    "externalise": "externalize",
    "externalised": "externalized",
    "externalises": "externalizes",
    "externalising": "externalizing",
    "factorise": "factorize",
    "factorised": "factorized",
    "factorises": "factorizes",
    "factorising": "factorizing",
    "faecal": "fecal",
    "faeces": "feces",
    "familiarisation": "familiarization",
    "familiarise": "familiarize",
    "familiarised": "familiarized",
    "familiarises": "familiarizes",
    "familiarising": "familiarizing",
    "fantasise": "fantasize",
    "fantasised": "fantasized",
    "fantasises": "fantasizes",
    "fantasising": "fantasizing",
    "favour": "favor",
    "favourable": "favorable",
    "favourably": "favorably",
    "favoured": "favored",
    "favouring": "favoring",
    "favourite": "favorite",
    "favourites": "favorites",
    "favouritism": "favoritism",
    "favours": "favors",
    "feminise": "feminize",
    "feminised": "feminized",
    "feminises": "feminizes",
    "feminising": "feminizing",
    "fertilisation": "fertilization",
    "fertilise": "fertilize",
    "fertilised": "fertilized",
    "fertiliser": "fertilizer",
    "fertilisers": "fertilizers",
    "fertilises": "fertilizes",
    "fertilising": "fertilizing",
    "fervour": "fervor",
    "fibre": "fiber",
    "fibreglass": "fiberglass",
    "fibres": "fibers",
    "fictionalisation": "fictionalization",
    "fictionalisations": "fictionalizations",
    "fictionalise": "fictionalize",
    "fictionalised": "fictionalized",
    "fictionalises": "fictionalizes",
    "fictionalising": "fictionalizing",
    "fillet": "filet",
    "filleted": "fileted",
    "filleting": "fileting",
    "fillets": "filets",
    "finalisation": "finalization",
    "finalise": "finalize",
    "finalised": "finalized",
    "finalises": "finalizes",
    "finalising": "finalizing",
    "flautist": "flutist",
    "flautists": "flutists",
    "flavour": "flavor",
    "flavoured": "flavored",
    "flavouring": "flavoring",
    "flavourings": "flavorings",
    "flavourless": "flavorless",
    "flavours": "flavors",
    "flavoursome": "flavorsome",
    "flyer / flier": "flier / flyer",
    "foetal": "fetal",
    "foetid": "fetid",
    "foetus": "fetus",
    "foetuses": "fetuses",
    "formalisation": "formalization",
    "formalise": "formalize",
    "formalised": "formalized",
    "formalises": "formalizes",
    "formalising": "formalizing",
    "fossilisation": "fossilization",
    "fossilise": "fossilize",
    "fossilised": "fossilized",
    "fossilises": "fossilizes",
    "fossilising": "fossilizing",
    "fraternisation": "fraternization",
    "fraternise": "fraternize",
    "fraternised": "fraternized",
    "fraternises": "fraternizes",
    "fraternising": "fraternizing",
    "fulfil": "fulfill",
    "fulfilment": "fulfillment",
    "fulfils": "fulfills",
    "funnelled": "funneled",
    "funnelling": "funneling",
    "gage": "gauge",
    "gaged": "gauged",
    "gages": "gauges",
    "gaging": "gauging",
    "galvanise": "galvanize",
    "galvanised": "galvanized",
    "galvanises": "galvanizes",
    "galvanising": "galvanizing",
    "gambolled": "gamboled",
    "gambolling": "gamboling",
    "gaol": "jail",
    "gaolbird": "jailbird",
    "gaolbirds": "jailbirds",
    "gaolbreak": "jailbreak",
    "gaolbreaks": "jailbreaks",
    "gaoled": "jailed",
    "gaoler": "jailer",
    "gaolers": "jailers",
    "gaoling": "jailing",
    "gaols": "jails",
    "gasses": "gases",
    "generalisation": "generalization",
    "generalisations": "generalizations",
    "generalise": "generalize",
    "generalised": "generalized",
    "generalises": "generalizes",
    "generalising": "generalizing",
    "ghettoise": "ghettoize",
    "ghettoised": "ghettoized",
    "ghettoises": "ghettoizes",
    "ghettoising": "ghettoizing",
    "gipsies": "gypsies",
    "glamor": "glamour",
    "glamorise": "glamorize",
    "glamorised": "glamorized",
    "glamorises": "glamorizes",
    "glamorising": "glamorizing",
    "globalisation": "globalization",
    "globalise": "globalize",
    "globalised": "globalized",
    "globalises": "globalizes",
    "globalising": "globalizing",
    "glueing": "gluing",
    "goitre": "goiter",
    "goitres": "goiters",
    "gonorrhoea": "gonorrhea",
    "gramme": "gram",
    "grammes": "grams",
    "gravelled": "graveled",
    "grey": "gray",
    "greyed": "grayed",
    "greying": "graying",
    "greyish": "grayish",
    "greyness": "grayness",
    "greys": "grays",
    "grovelled": "groveled",
    "grovelling": "groveling",
    "groyne": "groin",
    "groynes": "groins",
    "gruelling": "grueling",
    "gruellingly": "gruelingly",
    "gryphon": "griffin",
    "gryphons": "griffins",
    "gynaecological": "gynecological",
    "gynaecologist": "gynecologist",
    "gynaecologists": "gynecologists",
    "gynaecology": "gynecology",
    "haematological": "hematological",
    "haematologist": "hematologist",
    "haematologists": "hematologists",
    "haematology": "hematology",
    "haemoglobin": "hemoglobin",
    "haemophilia": "hemophilia",
    "haemophiliac": "hemophiliac",
    "haemophiliacs": "hemophiliacs",
    "haemorrhage": "hemorrhage",
    "haemorrhaged": "hemorrhaged",
    "haemorrhages": "hemorrhages",
    "haemorrhaging": "hemorrhaging",
    "haemorrhoids": "hemorrhoids",
    "harbour": "harbor",
    "harboured": "harbored",
    "harbouring": "harboring",
    "harbours": "harbors",
    "harmonisation": "harmonization",
    "harmonise": "harmonize",
    "harmonised": "harmonized",
    "harmonises": "harmonizes",
    "harmonising": "harmonizing",
    "homoeopath": "homeopath",
    "homoeopathic": "homeopathic",
    "homoeopaths": "homeopaths",
    "homoeopathy": "homeopathy",
    "homogenise": "homogenize",
    "homogenised": "homogenized",
    "homogenises": "homogenizes",
    "homogenising": "homogenizing",
    "honour": "honor",
    "honourable": "honorable",
    "honourably": "honorably",
    "honoured": "honored",
    "honouring": "honoring",
    "honours": "honors",
    "hospitalisation": "hospitalization",
    "hospitalise": "hospitalize",
    "hospitalised": "hospitalized",
    "hospitalises": "hospitalizes",
    "hospitalising": "hospitalizing",
    "humanise": "humanize",
    "humanised": "humanized",
    "humanises": "humanizes",
    "humanising": "humanizing",
    "humour": "humor",
    "humoured": "humored",
    "humouring": "humoring",
    "humourless": "humorless",
    "humours": "humors",
    "hybridise": "hybridize",
    "hybridised": "hybridized",
    "hybridises": "hybridizes",
    "hybridising": "hybridizing",
    "hypnotise": "hypnotize",
    "hypnotised": "hypnotized",
    "hypnotises": "hypnotizes",
    "hypnotising": "hypnotizing",
    "hypothesise": "hypothesize",
    "hypothesised": "hypothesized",
    "hypothesises": "hypothesizes",
    "hypothesising": "hypothesizing",
    "idealisation": "idealization",
    "idealise": "idealize",
    "idealised": "idealized",
    "idealises": "idealizes",
    "idealising": "idealizing",
    "idolise": "idolize",
    "idolised": "idolized",
    "idolises": "idolizes",
    "idolising": "idolizing",
    "immobilisation": "immobilization",
    "immobilise": "immobilize",
    "immobilised": "immobilized",
    "immobiliser": "immobilizer",
    "immobilisers": "immobilizers",
    "immobilises": "immobilizes",
    "immobilising": "immobilizing",
    "immortalise": "immortalize",
    "immortalised": "immortalized",
    "immortalises": "immortalizes",
    "immortalising": "immortalizing",
    "immunisation": "immunization",
    "immunise": "immunize",
    "immunised": "immunized",
    "immunises": "immunizes",
    "immunising": "immunizing",
    "impanelled": "impaneled",
    "impanelling": "impaneling",
    "imperilled": "imperiled",
    "imperilling": "imperiling",
    "individualise": "individualize",
    "individualised": "individualized",
    "individualises": "individualizes",
    "individualising": "individualizing",
    "industrialise": "industrialize",
    "industrialised": "industrialized",
    "industrialises": "industrializes",
    "industrialising": "industrializing",
    "inflexion": "inflection",
    "inflexions": "inflections",
    "initialise": "initialize",
    "initialised": "initialized",
    "initialises": "initializes",
    "initialising": "initializing",
    "initialled": "initialed",
    "initialling": "initialing",
    "instal": "install",
    "instalment": "installment",
    "instalments": "installments",
    "instals": "installs",
    "instil": "instill",
    "instils": "instills",
    "institutionalisation": "institutionalization",
    "institutionalise": "institutionalize",
    "institutionalised": "institutionalized",
    "institutionalises": "institutionalizes",
    "institutionalising": "institutionalizing",
    "intellectualise": "intellectualize",
    "intellectualised": "intellectualized",
    "intellectualises": "intellectualizes",
    "intellectualising": "intellectualizing",
    "internalisation": "internalization",
    "internalise": "internalize",
    "internalised": "internalized",
    "internalises": "internalizes",
    "internalising": "internalizing",
    "internationalisation": "internationalization",
    "internationalise": "internationalize",
    "internationalised": "internationalized",
    "internationalises": "internationalizes",
    "internationalising": "internationalizing",
    "ionisation": "ionization",
    "ionise": "ionize",
    "ionised": "ionized",
    "ioniser": "ionizer",
    "ionisers": "ionizers",
    "ionises": "ionizes",
    "ionising": "ionizing",
    "italicise": "italicize",
    "italicised": "italicized",
    "italicises": "italicizes",
    "italicising": "italicizing",
    "itemise": "itemize",
    "itemised": "itemized",
    "itemises": "itemizes",
    "itemising": "itemizing",
    "jeopardise": "jeopardize",
    "jeopardised": "jeopardized",
    "jeopardises": "jeopardizes",
    "jeopardising": "jeopardizing",
    "jewelled": "jeweled",
    "jeweller": "jeweler",
    "jewellers": "jewelers",
    "jewellery": "jewelry",
    "judgement": "judgment",
    "kilogramme": "kilogram",
    "kilogrammes": "kilograms",
    "kilometre": "kilometer",
    "kilometres": "kilometers",
    "labelled": "labeled",
    "labelling": "labeling",
    "labour": "labor",
    "laboured": "labored",
    "labourer": "laborer",
    "labourers": "laborers",
    "labouring": "laboring",
    "labours": "labors",
    "lacklustre": "lackluster",
    "legalisation": "legalization",
    "legalise": "legalize",
    "legalised": "legalized",
    "legalises": "legalizes",
    "legalising": "legalizing",
    "legitimise": "legitimize",
    "legitimised": "legitimized",
    "legitimises": "legitimizes",
    "legitimising": "legitimizing",
    "leukaemia": "leukemia",
    "levelled": "leveled",
    "leveller": "leveler",
    "levellers": "levelers",
    "levelling": "leveling",
    "libelled": "libeled",
    "libelling": "libeling",
    "libellous": "libelous",
    "liberalisation": "liberalization",
    "liberalise": "liberalize",
    "liberalised": "liberalized",
    "liberalises": "liberalizes",
    "liberalising": "liberalizing",
    "licence": "license",
    "licenced": "licensed",
    "licences": "licenses",
    "licencing": "licensing",
    "likeable": "likable",
    "lionisation": "lionization",
    "lionise": "lionize",
    "lionised": "lionized",
    "lionises": "lionizes",
    "lionising": "lionizing",
    "liquidise": "liquidize",
    "liquidised": "liquidized",
    "liquidiser": "liquidizer",
    "liquidisers": "liquidizers",
    "liquidises": "liquidizes",
    "liquidising": "liquidizing",
    "litre": "liter",
    "litres": "liters",
    "localise": "localize",
    "localised": "localized",
    "localises": "localizes",
    "localising": "localizing",
    "louvre": "louver",
    "louvred": "louvered",
    "louvres": "louvers",
    "lustre": "luster",
    "magnetise": "magnetize",
    "magnetised": "magnetized",
    "magnetises": "magnetizes",
    "magnetising": "magnetizing",
    "manoeuvrability": "maneuverability",
    "manoeuvrable": "maneuverable",
    "manoeuvre": "maneuver",
    "manoeuvred": "maneuvered",
    "manoeuvres": "maneuvers",
    "manoeuvring": "maneuvering",
    "manoeuvrings": "maneuverings",
    "marginalisation": "marginalization",
    "marginalise": "marginalize",
    "marginalised": "marginalized",
    "marginalises": "marginalizes",
    "marginalising": "marginalizing",
    "marshalled": "marshaled",
    "marshalling": "marshaling",
    "marvelled": "marveled",
    "marvelling": "marveling",
    "marvellous": "marvelous",
    "marvellously": "marvelously",
    "materialisation": "materialization",
    "materialise": "materialize",
    "materialised": "materialized",
    "materialises": "materializes",
    "materialising": "materializing",
    "maximisation": "maximization",
    "maximise": "maximize",
    "maximised": "maximized",
    "maximises": "maximizes",
    "maximising": "maximizing",
    "meagre": "meager",
    "mechanisation": "mechanization",
    "mechanise": "mechanize",
    "mechanised": "mechanized",
    "mechanises": "mechanizes",
    "mechanising": "mechanizing",
    "mediaeval": "medieval",
    "memorialise": "memorialize",
    "memorialised": "memorialized",
    "memorialises": "memorializes",
    "memorialising": "memorializing",
    "memorise": "memorize",
    "memorised": "memorized",
    "memorises": "memorizes",
    "memorising": "memorizing",
    "mesmerise": "mesmerize",
    "mesmerised": "mesmerized",
    "mesmerises": "mesmerizes",
    "mesmerising": "mesmerizing",
    "metabolise": "metabolize",
    "metabolised": "metabolized",
    "metabolises": "metabolizes",
    "metabolising": "metabolizing",
    "metre": "meter",
    "metres": "meters",
    "mhm": "hmm",
    "micrometre": "micrometer",
    "micrometres": "micrometers",
    "militarise": "militarize",
    "militarised": "militarized",
    "militarises": "militarizes",
    "militarising": "militarizing",
    "milligramme": "milligram",
    "milligrammes": "milligrams",
    "millilitre": "milliliter",
    "millilitres": "milliliters",
    "millimetre": "millimeter",
    "millimetres": "millimeters",
    "miniaturisation": "miniaturization",
    "miniaturise": "miniaturize",
    "miniaturised": "miniaturized",
    "miniaturises": "miniaturizes",
    "miniaturising": "miniaturizing",
    "minibusses": "minibuses",
    "minimise": "minimize",
    "minimised": "minimized",
    "minimises": "minimizes",
    "minimising": "minimizing",
    "misbehaviour": "misbehavior",
    "misdemeanour": "misdemeanor",
    "misdemeanours": "misdemeanors",
    "misspelt": "misspelled",
    "mitre": "miter",
    "mitres": "miters",
    "mm": "hmm",
    "mmm": "hmm",
    "mobilisation": "mobilization",
    "mobilise": "mobilize",
    "mobilised": "mobilized",
    "mobilises": "mobilizes",
    "mobilising": "mobilizing",
    "modelled": "modeled",
    "modeller": "modeler",
    "modellers": "modelers",
    "modelling": "modeling",
    "modernise": "modernize",
    "modernised": "modernized",
    "modernises": "modernizes",
    "modernising": "modernizing",
    "moisturise": "moisturize",
    "moisturised": "moisturized",
    "moisturiser": "moisturizer",
    "moisturisers": "moisturizers",
    "moisturises": "moisturizes",
    "moisturising": "moisturizing",
    "monologue": "monolog",
    "monologues": "monologs",
    "monopolisation": "monopolization",
    "monopolise": "monopolize",
    "monopolised": "monopolized",
    "monopolises": "monopolizes",
    "monopolising": "monopolizing",
    "moralise": "moralize",
    "moralised": "moralized",
    "moralises": "moralizes",
    "moralising": "moralizing",
    "motorised": "motorized",
    "mould": "mold",
    "moulded": "molded",
    "moulder": "molder",
    "mouldered": "moldered",
    "mouldering": "moldering",
    "moulders": "molders",
    "mouldier": "moldier",
    "mouldiest": "moldiest",
    "moulding": "molding",
    "mouldings": "moldings",
    "moulds": "molds",
    "mouldy": "moldy",
    "moult": "molt",
    "moulted": "molted",
    "moulting": "molting",
    "moults": "molts",
    "moustache": "mustache",
    "moustached": "mustached",
    "moustaches": "mustaches",
    "moustachioed": "mustachioed",
    "multicoloured": "multicolored",
    "nationalisation": "nationalization",
    "nationalisations": "nationalizations",
    "nationalise": "nationalize",
    "nationalised": "nationalized",
    "nationalises": "nationalizes",
    "nationalising": "nationalizing",
    "naturalisation": "naturalization",
    "naturalise": "naturalize",
    "naturalised": "naturalized",
    "naturalises": "naturalizes",
    "naturalising": "naturalizing",
    "neighbour": "neighbor",
    "neighbourhood": "neighborhood",
    "neighbourhoods": "neighborhoods",
    "neighbouring": "neighboring",
    "neighbourliness": "neighborliness",
    "neighbourly": "neighborly",
    "neighbours": "neighbors",
    "neutralisation": "neutralization",
    "neutralise": "neutralize",
    "neutralised": "neutralized",
    "neutralises": "neutralizes",
    "neutralising": "neutralizing",
    "normalisation": "normalization",
    "normalise": "normalize",
    "normalised": "normalized",
    "normalises": "normalizes",
    "normalising": "normalizing",
    "odour": "odor",
    "odourless": "odorless",
    "odours": "odors",
    "oesophagus": "esophagus",
    "oesophaguses": "esophaguses",
    "oestrogen": "estrogen",
    "offence": "offense",
    "offences": "offenses",
    "omelette": "omelet",
    "omelettes": "omelets",
    "optimise": "optimize",
    "optimised": "optimized",
    "optimises": "optimizes",
    "optimising": "optimizing",
    "organisation": "organization",
    "organisational": "organizational",
    "organisations": "organizations",
    "organise": "organize",
    "organised": "organized",
    "organiser": "organizer",
    "organisers": "organizers",
    "organises": "organizes",
    "organising": "organizing",
    "orthopaedic": "orthopedic",
    "orthopaedics": "orthopedics",
    "ostracise": "ostracize",
    "ostracised": "ostracized",
    "ostracises": "ostracizes",
    "ostracising": "ostracizing",
    "outmanoeuvre": "outmaneuver",
    "outmanoeuvred": "outmaneuvered",
    "outmanoeuvres": "outmaneuvers",
    "outmanoeuvring": "outmaneuvering",
    "overemphasise": "overemphasize",
    "overemphasised": "overemphasized",
    "overemphasises": "overemphasizes",
    "overemphasising": "overemphasizing",
    "oxidisation": "oxidization",
    "oxidise": "oxidize",
    "oxidised": "oxidized",
    "oxidises": "oxidizes",
    "oxidising": "oxidizing",
    "paederast": "pederast",
    "paederasts": "pederasts",
    "paediatric": "pediatric",
    "paediatrician": "pediatrician",
    "paediatricians": "pediatricians",
    "paediatrics": "pediatrics",
    "paedophile": "pedophile",
    "paedophiles": "pedophiles",
    "paedophilia": "pedophilia",
    "palaeolithic": "paleolithic",
    "palaeontologist": "paleontologist",
    "palaeontologists": "paleontologists",
    "palaeontology": "paleontology",
    "panelled": "paneled",
    "panelling": "paneling",
    "panellist": "panelist",
    "panellists": "panelists",
    "paralyse": "paralyze",
    "paralysed": "paralyzed",
    "paralyses": "paralyzes",
    "paralysing": "paralyzing",
    "parcelled": "parceled",
    "parcelling": "parceling",
    "parlour": "parlor",
    "parlours": "parlors",
    "particularise": "particularize",
    "particularised": "particularized",
    "particularises": "particularizes",
    "particularising": "particularizing",
    "passivisation": "passivization",
    "passivise": "passivize",
    "passivised": "passivized",
    "passivises": "passivizes",
    "passivising": "passivizing",
    "pasteurisation": "pasteurization",
    "pasteurise": "pasteurize",
    "pasteurised": "pasteurized",
    "pasteurises": "pasteurizes",
    "pasteurising": "pasteurizing",
    "patronise": "patronize",
    "patronised": "patronized",
    "patronises": "patronizes",
    "patronising": "patronizing",
    "patronisingly": "patronizingly",
    "pedalled": "pedaled",
    "pedalling": "pedaling",
    "pedestrianisation": "pedestrianization",
    "pedestrianise": "pedestrianize",
    "pedestrianised": "pedestrianized",
    "pedestrianises": "pedestrianizes",
    "pedestrianising": "pedestrianizing",
    "penalise": "penalize",
    "penalised": "penalized",
    "penalises": "penalizes",
    "penalising": "penalizing",
    "pencilled": "penciled",
    "pencilling": "penciling",
    "personalise": "personalize",
    "personalised": "personalized",
    "personalises": "personalizes",
    "personalising": "personalizing",
    "pharmacopoeia": "pharmacopeia",
    "pharmacopoeias": "pharmacopeias",
    "philosophise": "philosophize",
    "philosophised": "philosophized",
    "philosophises": "philosophizes",
    "philosophising": "philosophizing",
    "philtre": "filter",
    "philtres": "filters",
    "phoney": "phony",
    "plagiarise": "plagiarize",
    "plagiarised": "plagiarized",
    "plagiarises": "plagiarizes",
    "plagiarising": "plagiarizing",
    "plough": "plow",
    "ploughed": "plowed",
    "ploughing": "plowing",
    "ploughman": "plowman",
    "ploughmen": "plowmen",
    "ploughs": "plows",
    "ploughshare": "plowshare",
    "ploughshares": "plowshares",
    "polarisation": "polarization",
    "polarise": "polarize",
    "polarised": "polarized",
    "polarises": "polarizes",
    "polarising": "polarizing",
    "politicisation": "politicization",
    "politicise": "politicize",
    "politicised": "politicized",
    "politicises": "politicizes",
    "politicising": "politicizing",
    "popularisation": "popularization",
    "popularise": "popularize",
    "popularised": "popularized",
    "popularises": "popularizes",
    "popularising": "popularizing",
    "pouffe": "pouf",
    "pouffes": "poufs",
    "practise": "practice",
    "practised": "practiced",
    "practises": "practices",
    "practising": "practicing",
    "praesidium": "presidium",
    "praesidiums": "presidiums",
    "pressurisation": "pressurization",
    "pressurise": "pressurize",
    "pressurised": "pressurized",
    "pressurises": "pressurizes",
    "pressurising": "pressurizing",
    "pretence": "pretense",
    "pretences": "pretenses",
    "primaeval": "primeval",
    "prioritisation": "prioritization",
    "prioritise": "prioritize",
    "prioritised": "prioritized",
    "prioritises": "prioritizes",
    "prioritising": "prioritizing",
    "privatisation": "privatization",
    "privatisations": "privatizations",
    "privatise": "privatize",
    "privatised": "privatized",
    "privatises": "privatizes",
    "privatising": "privatizing",
    "professionalisation": "professionalization",
    "professionalise": "professionalize",
    "professionalised": "professionalized",
    "professionalises": "professionalizes",
    "professionalising": "professionalizing",
    "programme": "program",
    "programmes": "programs",
    "prologue": "prolog",
    "prologues": "prologs",
    "propagandise": "propagandize",
    "propagandised": "propagandized",
    "propagandises": "propagandizes",
    "propagandising": "propagandizing",
    "proselytise": "proselytize",
    "proselytised": "proselytized",
    "proselytiser": "proselytizer",
    "proselytisers": "proselytizers",
    "proselytises": "proselytizes",
    "proselytising": "proselytizing",
    "psychoanalyse": "psychoanalyze",
    "psychoanalysed": "psychoanalyzed",
    "psychoanalyses": "psychoanalyzes",
    "psychoanalysing": "psychoanalyzing",
    "publicise": "publicize",
    "publicised": "publicized",
    "publicises": "publicizes",
    "publicising": "publicizing",
    "pulverisation": "pulverization",
    "pulverise": "pulverize",
    "pulverised": "pulverized",
    "pulverises": "pulverizes",
    "pulverising": "pulverizing",
    "pummelled": "pummel",
    "pummelling": "pummeled",
    "pyjama": "pajama",
    "pyjamas": "pajamas",
    "pzazz": "pizzazz",
    "quarrelled": "quarreled",
    "quarrelling": "quarreling",
    "radicalise": "radicalize",
    "radicalised": "radicalized",
    "radicalises": "radicalizes",
    "radicalising": "radicalizing",
    "rancour": "rancor",
    "randomise": "randomize",
    "randomised": "randomized",
    "randomises": "randomizes",
    "randomising": "randomizing",
    "rationalisation": "rationalization",
    "rationalisations": "rationalizations",
    "rationalise": "rationalize",
    "rationalised": "rationalized",
    "rationalises": "rationalizes",
    "rationalising": "rationalizing",
    "ravelled": "raveled",
    "ravelling": "raveling",
    "realisable": "realizable",
    "realisation": "realization",
    "realisations": "realizations",
    "realise": "realize",
    "realised": "realized",
    "realises": "realizes",
    "realising": "realizing",
    "recognisable": "recognizable",
    "recognisably": "recognizably",
    "recognisance": "recognizance",
    "recognise": "recognize",
    "recognised": "recognized",
    "recognises": "recognizes",
    "recognising": "recognizing",
    "reconnoitre": "reconnoiter",
    "reconnoitred": "reconnoitered",
    "reconnoitres": "reconnoiters",
    "reconnoitring": "reconnoitering",
    "refuelled": "refueled",
    "refuelling": "refueling",
    "regularisation": "regularization",
    "regularise": "regularize",
    "regularised": "regularized",
    "regularises": "regularizes",
    "regularising": "regularizing",
    "remodelled": "remodeled",
    "remodelling": "remodeling",
    "remould": "remold",
    "remoulded": "remolded",
    "remoulding": "remolding",
    "remoulds": "remolds",
    "reorganisation": "reorganization",
    "reorganisations": "reorganizations",
    "reorganise": "reorganize",
    "reorganised": "reorganized",
    "reorganises": "reorganizes",
    "reorganising": "reorganizing",
    "revelled": "reveled",
    "reveller": "reveler",
    "revellers": "revelers",
    "revelling": "reveling",
    "revitalise": "revitalize",
    "revitalised": "revitalized",
    "revitalises": "revitalizes",
    "revitalising": "revitalizing",
    "revolutionise": "revolutionize",
    "revolutionised": "revolutionized",
    "revolutionises": "revolutionizes",
    "revolutionising": "revolutionizing",
    "rhapsodise": "rhapsodize",
    "rhapsodised": "rhapsodized",
    "rhapsodises": "rhapsodizes",
    "rhapsodising": "rhapsodizing",
    "rigour": "rigor",
    "rigours": "rigors",
    "ritualised": "ritualized",
    "rivalled": "rivaled",
    "rivalling": "rivaling",
    "romanticise": "romanticize",
    "romanticised": "romanticized",
    "romanticises": "romanticizes",
    "romanticising": "romanticizing",
    "rumour": "rumor",
    "rumoured": "rumored",
    "rumours": "rumors",
    "sabre": "saber",
    "sabres": "sabers",
    "saltpetre": "saltpeter",
    "sanitise": "sanitize",
    "sanitised": "sanitized",
    "sanitises": "sanitizes",
    "sanitising": "sanitizing",
    "satirise": "satirize",
    "satirised": "satirized",
    "satirises": "satirizes",
    "satirising": "satirizing",
    "saviour": "savior",
    "saviours": "saviors",
    "savour": "savor",
    "savoured": "savored",
    "savouries": "savories",
    "savouring": "savoring",
    "savours": "savors",
    "savoury": "savory",
    "scandalise": "scandalize",
    "scandalised": "scandalized",
    "scandalises": "scandalizes",
    "scandalising": "scandalizing",
    "sceptic": "skeptic",
    "sceptical": "skeptical",
    "sceptically": "skeptically",
    "scepticism": "skepticism",
    "sceptics": "skeptics",
    "sceptre": "scepter",
    "sceptres": "scepters",
    "scrutinise": "scrutinize",
    "scrutinised": "scrutinized",
    "scrutinises": "scrutinizes",
    "scrutinising": "scrutinizing",
    "secularisation": "secularization",
    "secularise": "secularize",
    "secularised": "secularized",
    "secularises": "secularizes",
    "secularising": "secularizing",
    "sensationalise": "sensationalize",
    "sensationalised": "sensationalized",
    "sensationalises": "sensationalizes",
    "sensationalising": "sensationalizing",
    "sensitise": "sensitize",
    "sensitised": "sensitized",
    "sensitises": "sensitizes",
    "sensitising": "sensitizing",
    "sentimentalise": "sentimentalize",
    "sentimentalised": "sentimentalized",
    "sentimentalises": "sentimentalizes",
    "sentimentalising": "sentimentalizing",
    "sepulchre": "sepulcher",
    "sepulchres": "sepulchers",
    "serialisation": "serialization",
    "serialisations": "serializations",
    "serialise": "serialize",
    "serialised": "serialized",
    "serialises": "serializes",
    "serialising": "serializing",
    "sermonise": "sermonize",
    "sermonised": "sermonized",
    "sermonises": "sermonizes",
    "sermonising": "sermonizing",
    "sheikh": "sheik",
    "shovelled": "shoveled",
    "shovelling": "shoveling",
    "shrivelled": "shriveled",
    "shrivelling": "shriveling",
    "signalise": "signalize",
    "signalised": "signalized",
    "signalises": "signalizes",
    "signalising": "signalizing",
    "signalled": "signaled",
    "signalling": "signaling",
    "smoulder": "smolder",
    "smouldered": "smoldered",
    "smouldering": "smoldering",
    "smoulders": "smolders",
    "snivelled": "sniveled",
    "snivelling": "sniveling",
    "snorkelled": "snorkeled",
    "snorkelling": "snorkeling",
    "snowplough": "snowplow",
    "snowploughs": "snowplow",
    "socialisation": "socialization",
    "socialise": "socialize",
    "socialised": "socialized",
    "socialises": "socializes",
    "socialising": "socializing",
    "sodomise": "sodomize",
    "sodomised": "sodomized",
    "sodomises": "sodomizes",
    "sodomising": "sodomizing",
    "solemnise": "solemnize",
    "solemnised": "solemnized",
    "solemnises": "solemnizes",
    "solemnising": "solemnizing",
    "sombre": "somber",
    "specialisation": "specialization",
    "specialisations": "specializations",
    "specialise": "specialize",
    "specialised": "specialized",
    "specialises": "specializes",
    "specialising": "specializing",
    "spectre": "specter",
    "spectres": "specters",
    "spiralled": "spiraled",
    "spiralling": "spiraling",
    "splendour": "splendor",
    "splendours": "splendors",
    "squirrelled": "squirreled",
    "squirrelling": "squirreling",
    "stabilisation": "stabilization",
    "stabilise": "stabilize",
    "stabilised": "stabilized",
    "stabiliser": "stabilizer",
    "stabilisers": "stabilizers",
    "stabilises": "stabilizes",
    "stabilising": "stabilizing",
    "standardisation": "standardization",
    "standardise": "standardize",
    "standardised": "standardized",
    "standardises": "standardizes",
    "standardising": "standardizing",
    "stencilled": "stenciled",
    "stencilling": "stenciling",
    "sterilisation": "sterilization",
    "sterilisations": "sterilizations",
    "sterilise": "sterilize",
    "sterilised": "sterilized",
    "steriliser": "sterilizer",
    "sterilisers": "sterilizers",
    "sterilises": "sterilizes",
    "sterilising": "sterilizing",
    "stigmatisation": "stigmatization",
    "stigmatise": "stigmatize",
    "stigmatised": "stigmatized",
    "stigmatises": "stigmatizes",
    "stigmatising": "stigmatizing",
    "storey": "story",
    "storeys": "stories",
    "subsidisation": "subsidization",
    "subsidise": "subsidize",
    "subsidised": "subsidized",
    "subsidiser": "subsidizer",
    "subsidisers": "subsidizers",
    "subsidises": "subsidizes",
    "subsidising": "subsidizing",
    "succour": "succor",
    "succoured": "succored",
    "succouring": "succoring",
    "succours": "succors",
    "sulphate": "sulfate",
    "sulphates": "sulfates",
    "sulphide": "sulfide",
    "sulphides": "sulfides",
    "sulphur": "sulfur",
    "sulphurous": "sulfurous",
    "summarise": "summarize",
    "summarised": "summarized",
    "summarises": "summarizes",
    "summarising": "summarizing",
    "swivelled": "swiveled",
    "swivelling": "swiveling",
    "symbolise": "symbolize",
    "symbolised": "symbolized",
    "symbolises": "symbolizes",
    "symbolising": "symbolizing",
    "sympathise": "sympathize",
    "sympathised": "sympathized",
    "sympathiser": "sympathizer",
    "sympathisers": "sympathizers",
    "sympathises": "sympathizes",
    "sympathising": "sympathizing",
    "synchronisation": "synchronization",
    "synchronise": "synchronize",
    "synchronised": "synchronized",
    "synchronises": "synchronizes",
    "synchronising": "synchronizing",
    "synthesise": "synthesize",
    "synthesised": "synthesized",
    "synthesiser": "synthesizer",
    "synthesisers": "synthesizers",
    "synthesises": "synthesizes",
    "synthesising": "synthesizing",
    "syphon": "siphon",
    "syphoned": "siphoned",
    "syphoning": "siphoning",
    "syphons": "siphons",
    "systematisation": "systematization",
    "systematise": "systematize",
    "systematised": "systematized",
    "systematises": "systematizes",
    "systematising": "systematizing",
    "tantalise": "tantalize",
    "tantalised": "tantalized",
    "tantalises": "tantalizes",
    "tantalising": "tantalizing",
    "tantalisingly": "tantalizingly",
    "tasselled": "tasseled",
    "technicolour": "technicolor",
    "temporise": "temporize",
    "temporised": "temporized",
    "temporises": "temporizes",
    "temporising": "temporizing",
    "tenderise": "tenderize",
    "tenderised": "tenderized",
    "tenderises": "tenderizes",
    "tenderising": "tenderizing",
    "terrorise": "terrorize",
    "terrorised": "terrorized",
    "terrorises": "terrorizes",
    "terrorising": "terrorizing",
    "theatre": "theater",
    "theatregoer": "theatergoer",
    "theatregoers": "theatergoers",
    "theatres": "theaters",
    "theorise": "theorize",
    "theorised": "theorized",
    "theorises": "theorizes",
    "theorising": "theorizing",
    "tonne": "ton",
    "tonnes": "tons",
    "towelled": "toweled",
    "towelling": "toweling",
    "toxaemia": "toxemia",
    "tranquillise": "tranquilize",
    "tranquillised": "tranquilized",
    "tranquilliser": "tranquilizer",
    "tranquillisers": "tranquilizers",
    "tranquillises": "tranquilizes",
    "tranquillising": "tranquilizing",
    "tranquillity": "tranquility",
    "tranquillize": "tranquilize",
    "tranquillized": "tranquilized",
    "tranquillizer": "tranquilizer",
    "tranquillizers": "tranquilizers",
    "tranquillizes": "tranquilizes",
    "tranquillizing": "tranquilizing",
    "tranquilly": "tranquility",
    "transistorised": "transistorized",
    "traumatise": "traumatize",
    "traumatised": "traumatized",
    "traumatises": "traumatizes",
    "traumatising": "traumatizing",
    "travelled": "traveled",
    "traveller": "traveler",
    "travellers": "travelers",
    "travelling": "traveling",
    "travelog": "travelogue",
    "travelogs": "travelogues",
    "trialled": "trialed",
    "trialling": "trialing",
    "tricolour": "tricolor",
    "tricolours": "tricolors",
    "trivialise": "trivialize",
    "trivialised": "trivialized",
    "trivialises": "trivializes",
    "trivialising": "trivializing",
    "tumour": "tumor",
    "tumours": "tumors",
    "tunnelled": "tunneled",
    "tunnelling": "tunneling",
    "tyrannise": "tyrannize",
    "tyrannised": "tyrannized",
    "tyrannises": "tyrannizes",
    "tyrannising": "tyrannizing",
    "tyre": "tire",
    "tyres": "tires",
    "unauthorised": "unauthorized",
    "uncivilised": "uncivilized",
    "underutilised": "underutilized",
    "unequalled": "unequaled",
    "unfavourable": "unfavorable",
    "unfavourably": "unfavorably",
    "unionisation": "unionization",
    "unionise": "unionize",
    "unionised": "unionized",
    "unionises": "unionizes",
    "unionising": "unionizing",
    "unorganised": "unorganized",
    "unravelled": "unraveled",
    "unravelling": "unraveling",
    "unrecognisable": "unrecognizable",
    "unrecognised": "unrecognized",
    "unrivalled": "unrivaled",
    "unsavoury": "unsavory",
    "untrammelled": "untrammeled",
    "urbanisation": "urbanization",
    "urbanise": "urbanize",
    "urbanised": "urbanized",
    "urbanises": "urbanizes",
    "urbanising": "urbanizing",
    "utilisable": "utilizable",
    "utilisation": "utilization",
    "utilise": "utilize",
    "utilised": "utilized",
    "utilises": "utilizes",
    "utilising": "utilizing",
    "valour": "valor",
    "vandalise": "vandalize",
    "vandalised": "vandalized",
    "vandalises": "vandalizes",
    "vandalising": "vandalizing",
    "vaporisation": "vaporization",
    "vaporise": "vaporize",
    "vaporised": "vaporized",
    "vaporises": "vaporizes",
    "vaporising": "vaporizing",
    "vapour": "vapor",
    "vapours": "vapors",
    "verbalise": "verbalize",
    "verbalised": "verbalized",
    "verbalises": "verbalizes",
    "verbalising": "verbalizing",
    "victimisation": "victimization",
    "victimise": "victimize",
    "victimised": "victimized",
    "victimises": "victimizes",
    "victimising": "victimizing",
    "videodisc": "videodisk",
    "videodiscs": "videodisks",
    "vigour": "vigor",
    "visualisation": "visualization",
    "visualisations": "visualizations",
    "visualise": "visualize",
    "visualised": "visualized",
    "visualises": "visualizes",
    "visualising": "visualizing",
    "vocalisation": "vocalization",
    "vocalisations": "vocalizations",
    "vocalise": "vocalize",
    "vocalised": "vocalized",
    "vocalises": "vocalizes",
    "vocalising": "vocalizing",
    "vulcanised": "vulcanized",
    "vulgarisation": "vulgarization",
    "vulgarise": "vulgarize",
    "vulgarised": "vulgarized",
    "vulgarises": "vulgarizes",
    "vulgarising": "vulgarizing",
    "waggon": "wagon",
    "waggons": "wagons",
    "watercolour": "watercolor",
    "watercolours": "watercolors",
    "weaselled": "weaseled",
    "weaselling": "weaseling",
    "westernisation": "westernization",
    "westernise": "westernize",
    "westernised": "westernized",
    "westernises": "westernizes",
    "westernising": "westernizing",
    "womanise": "womanize",
    "womanised": "womanized",
    "womaniser": "womanizer",
    "womanisers": "womanizers",
    "womanises": "womanizes",
    "womanising": "womanizing",
    "woollen": "woolen",
    "woollens": "woolens",
    "woollies": "woolies",
    "woolly": "wooly",
    "worshipped": "worshiped",
    "worshipper": "worshiper",
    "worshipping": "worshiping",
    "yodelled": "yodeled",
    "yodelling": "yodeling",
    "yoghourt": "yogurt",
    "yoghourts": "yogurts",
    "yoghurt": "yogurt",
    "yoghurts": "yogurts",
}


def remove_symbols_and_diacritics(s: str, keep=""):
    """
    Replace any other markers, symbols, and punctuations with a space,
    and drop any diacritics (category 'Mn' and some manual mappings)
    """

    def replace_character(char):
        if char in keep:
            return char
        elif char in ADDITIONAL_DIACRITICS:
            return ADDITIONAL_DIACRITICS[char]
        elif unicodedata.category(char) == "Mn":
            return ""
        elif unicodedata.category(char)[0] in "MSP":
            return " "
        return char

    return "".join(replace_character(c) for c in unicodedata.normalize("NFKD", s))


def remove_symbols(s: str):
    """
    Replace any other markers, symbols, punctuations with a space,
    keeping diacritics
    """
    return "".join(" " if unicodedata.category(c)[0] in "MSP" else c for c in unicodedata.normalize("NFKC", s))


class BasicMultilingualTextNormalizer:
    """
    Basic multilingual text normalizer for non-English languages.
    Removes diacritics, punctuation, and normalizes whitespace.
    """

    def __init__(self, remove_diacritics: bool = True):
        self.clean = remove_symbols_and_diacritics if remove_diacritics else remove_symbols

    def __call__(self, s: str):
        # FIRST: Remove chat template role markers (before any other processing)
        # This ensures accurate WER/CER by excluding prompt tokens from comparison
        s = remove_chat_template_roles(s)

        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = self.clean(s).lower()

        # Remove punctuations and extra spaces
        s = re.sub(r"[^\w\s]", "", s)
        s = re.sub(r"\s+", " ", s).strip()

        return s


class EnglishNumberNormalizer:
    """
    Convert any spelled-out numbers into arabic numbers, handling:
    - Remove commas
    - Keep suffixes: `1960s`, `274th`, `32nd`, etc.
    - Spell out currency symbols: `$20 million` -> `20000000 dollars`
    - Spell out `one` and `ones`
    - Interpret successive single-digit numbers: `one oh one` -> `101`
    """

    def __init__(self):
        super().__init__()

        self.zeros = {"o", "oh", "zero"}
        # fmt: off
        self.ones = {
            name: i
            for i, name in enumerate(
                ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                 "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                 "seventeen", "eighteen", "nineteen"],
                start=1,
            )
        }
        # fmt: on
        self.ones_plural = {
            "sixes" if name == "six" else name + "s": (value, "s") for name, value in self.ones.items()
        }
        self.ones_ordinal = {
            "zeroth": (0, "th"),
            "first": (1, "st"),
            "second": (2, "nd"),
            "third": (3, "rd"),
            "fifth": (5, "th"),
            "twelfth": (12, "th"),
            **{
                name + ("h" if name.endswith("t") else "th"): (value, "th")
                for name, value in self.ones.items()
                if value > 3 and value != 5 and value != 12
            },
        }
        self.ones_suffixed = {**self.ones_plural, **self.ones_ordinal}

        self.tens = {
            "twenty": 20,
            "thirty": 30,
            "forty": 40,
            "fifty": 50,
            "sixty": 60,
            "seventy": 70,
            "eighty": 80,
            "ninety": 90,
        }
        self.tens_plural = {name.replace("y", "ies"): (value, "s") for name, value in self.tens.items()}
        self.tens_ordinal = {name.replace("y", "ieth"): (value, "th") for name, value in self.tens.items()}
        self.tens_suffixed = {**self.tens_plural, **self.tens_ordinal}

        self.multipliers = {
            "hundred": 100,
            "thousand": 1_000,
            "million": 1_000_000,
            "billion": 1_000_000_000,
            "trillion": 1_000_000_000_000,
            "quadrillion": 1_000_000_000_000_000,
            "quintillion": 1_000_000_000_000_000_000,
            "sextillion": 1_000_000_000_000_000_000_000,
            "septillion": 1_000_000_000_000_000_000_000_000,
            "octillion": 1_000_000_000_000_000_000_000_000_000,
            "nonillion": 1_000_000_000_000_000_000_000_000_000_000,
            "decillion": 1_000_000_000_000_000_000_000_000_000_000_000,
        }
        self.multipliers_plural = {name + "s": (value, "s") for name, value in self.multipliers.items()}
        self.multipliers_ordinal = {name + "th": (value, "th") for name, value in self.multipliers.items()}
        self.multipliers_suffixed = {**self.multipliers_plural, **self.multipliers_ordinal}
        self.decimals = {*self.ones, *self.tens, *self.zeros}

        self.preceding_prefixers = {
            "minus": "-",
            "negative": "-",
            "plus": "+",
            "positive": "+",
        }
        self.following_prefixers = {
            "pound": "£",
            "pounds": "£",
            "euro": "€",
            "euros": "€",
            "dollar": "$",
            "dollars": "$",
            "cent": "¢",
            "cents": "¢",
        }
        self.prefixes = set(list(self.preceding_prefixers.values()) + list(self.following_prefixers.values()))
        self.suffixers = {
            "per": {"cent": "%"},
            "percent": "%",
        }
        self.specials = {"and", "double", "triple", "point"}

        self.words = {
            key
            for mapping in [
                self.zeros,
                self.ones,
                self.ones_suffixed,
                self.tens,
                self.tens_suffixed,
                self.multipliers,
                self.multipliers_suffixed,
                self.preceding_prefixers,
                self.following_prefixers,
                self.suffixers,
                self.specials,
            ]
            for key in mapping
        }
        self.literal_words = {"one", "ones"}

    def process_words(self, words: List[str]) -> Iterator[str]:
        prefix: Optional[str] = None
        value: Optional[Union[str, int]] = None
        skip = False

        def to_fraction(s: str):
            try:
                return Fraction(s)
            except ValueError:
                return None

        def output(result: Union[str, int]):
            nonlocal prefix, value
            result = str(result)
            if prefix is not None:
                result = prefix + result
            value = None
            prefix = None
            return result

        if len(words) == 0:
            return

        for i, current in enumerate(words):
            prev = words[i - 1] if i != 0 else None
            next = words[i + 1] if i != len(words) - 1 else None
            if skip:
                skip = False
                continue

            next_is_numeric = next is not None and re.match(r"^\d+(\.\d+)?$", next)
            has_prefix = current[0] in self.prefixes
            current_without_prefix = current[1:] if has_prefix else current

            if re.match(r"^\d+(\.\d+)?$", current_without_prefix):
                # Arabic numbers (potentially with signs and fractions)
                f = to_fraction(current_without_prefix)
                if f is None:
                    raise ValueError("Converting the fraction failed")

                if value is not None:
                    if isinstance(value, str) and value.endswith("."):
                        # Concatenate decimals / ip address components
                        value = str(value) + str(current)
                        continue
                    else:
                        yield output(value)

                prefix = current[0] if has_prefix else prefix
                if f.denominator == 1:
                    value = f.numerator  # Store integers as int
                else:
                    value = current_without_prefix

            elif current not in self.words:
                # Non-numeric words
                if value is not None:
                    yield output(value)
                yield output(current)

            elif current in self.zeros:
                value = str(value or "") + "0"

            elif current in self.ones:
                ones = self.ones[current]

                if value is None:
                    value = ones
                elif isinstance(value, str) or prev in self.ones:
                    if prev in self.tens and ones < 10:  # Replace the last zero with the digit
                        value = value[:-1] + str(ones)
                    else:
                        value = str(value) + str(ones)
                elif ones < 10:
                    if value % 10 == 0:
                        value += ones
                    else:
                        value = str(value) + str(ones)
                else:  # Eleven to nineteen
                    if value % 100 == 0:
                        value += ones
                    else:
                        value = str(value) + str(ones)

            elif current in self.ones_suffixed:
                # Ordinal or cardinal; yield the number right away
                ones, suffix = self.ones_suffixed[current]
                if value is None:
                    yield output(str(ones) + suffix)
                elif isinstance(value, str) or prev in self.ones:
                    if prev in self.tens and ones < 10:
                        yield output(value[:-1] + str(ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                elif ones < 10:
                    if value % 10 == 0:
                        yield output(str(value + ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                else:  # Eleven to nineteen
                    if value % 100 == 0:
                        yield output(str(value + ones) + suffix)
                    else:
                        yield output(str(value) + str(ones) + suffix)
                value = None

            elif current in self.tens:
                tens = self.tens[current]
                if value is None:
                    value = tens
                elif isinstance(value, str):
                    value = str(value) + str(tens)
                else:
                    if value % 100 == 0:
                        value += tens
                    else:
                        value = str(value) + str(tens)

            elif current in self.tens_suffixed:
                # Ordinal or cardinal; yield the number right away
                tens, suffix = self.tens_suffixed[current]
                if value is None:
                    yield output(str(tens) + suffix)
                elif isinstance(value, str):
                    yield output(str(value) + str(tens) + suffix)
                else:
                    if value % 100 == 0:
                        yield output(str(value + tens) + suffix)
                    else:
                        yield output(str(value) + str(tens) + suffix)

            elif current in self.multipliers:
                multiplier = self.multipliers[current]
                if value is None:
                    value = multiplier
                elif isinstance(value, str) or value == 0:
                    f = to_fraction(value)
                    p = f * multiplier if f is not None else None
                    if f is not None and p.denominator == 1:
                        value = p.numerator
                    else:
                        yield output(value)
                        value = multiplier
                else:
                    before = value // 1000 * 1000
                    residual = value % 1000
                    value = before + residual * multiplier

            elif current in self.multipliers_suffixed:
                multiplier, suffix = self.multipliers_suffixed[current]
                if value is None:
                    yield output(str(multiplier) + suffix)
                elif isinstance(value, str):
                    f = to_fraction(value)
                    p = f * multiplier if f is not None else None
                    if f is not None and p.denominator == 1:
                        yield output(str(p.numerator) + suffix)
                    else:
                        yield output(value)
                        yield output(str(multiplier) + suffix)
                else:  # Int
                    before = value // 1000 * 1000
                    residual = value % 1000
                    value = before + residual * multiplier
                    yield output(str(value) + suffix)
                value = None

            elif current in self.preceding_prefixers:
                # Apply prefix (positive, minus, etc.) if it precedes a number
                if value is not None:
                    yield output(value)

                if next in self.words or next_is_numeric:
                    prefix = self.preceding_prefixers[current]
                else:
                    yield output(current)

            elif current in self.following_prefixers:
                # Apply prefix (dollars, cents, etc.) only after a number
                if value is not None:
                    prefix = self.following_prefixers[current]
                    yield output(value)
                else:
                    yield output(current)

            elif current in self.suffixers:
                # Apply suffix symbols (percent -> '%')
                if value is not None:
                    suffix = self.suffixers[current]
                    if isinstance(suffix, dict):
                        if next in suffix:
                            yield output(str(value) + suffix[next])
                            skip = True
                        else:
                            yield output(value)
                            yield output(current)
                    else:
                        yield output(str(value) + suffix)
                else:
                    yield output(current)

            elif current in self.specials:
                if next not in self.words and not next_is_numeric:
                    # Apply special handling only if the next word can be numeric
                    if value is not None:
                        yield output(value)
                    yield output(current)
                elif current == "and":
                    # Ignore "and" after hundreds, thousands, etc.
                    if prev not in self.multipliers:
                        if value is not None:
                            yield output(value)
                        yield output(current)
                elif current == "double" or current == "triple":
                    if next in self.ones or next in self.zeros:
                        repeats = 2 if current == "double" else 3
                        ones = self.ones.get(next, 0)
                        value = str(value or "") + str(ones) * repeats
                        skip = True
                    else:
                        if value is not None:
                            yield output(value)
                        yield output(current)
                elif current == "point":
                    if next in self.decimals or next_is_numeric:
                        value = str(value or "") + "."
                else:
                    # Should all have been covered at this point
                    raise ValueError(f"Unexpected token: {current}")
            else:
                # All should have been covered at this point
                raise ValueError(f"Unexpected token: {current}")

        if value is not None:
            yield output(value)

    def preprocess(self, s: str):
        # Replace "<number> and a half" with "<number> point five"
        results = []

        segments = re.split(r"\band\s+a\s+half\b", s)
        for i, segment in enumerate(segments):
            if len(segment.strip()) == 0:
                continue
            if i == len(segments) - 1:
                results.append(segment)
            else:
                results.append(segment)
                last_word = segment.rsplit(maxsplit=2)[-1]
                if last_word in self.decimals or last_word in self.multipliers:
                    results.append("point five")
                else:
                    results.append("and a half")

        s = " ".join(results)

        # Put a space at number/letter boundary
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)

        # But remove spaces which could be a suffix
        s = re.sub(r"([0-9])\s+(st|nd|rd|th|s)\b", r"\1\2", s)

        return s

    def postprocess(self, s: str):
        def combine_cents(m: Match):
            try:
                currency = m.group(1)
                integer = m.group(2)
                cents = int(m.group(3))
                return f"{currency}{integer}.{cents:02d}"
            except ValueError:
                return m.string

        def extract_cents(m: Match):
            try:
                return f"¢{int(m.group(1))}"
            except ValueError:
                return m.string

        # Apply currency postprocessing; "$2 and ¢7" -> "$2.07"
        s = re.sub(r"([€£$])([0-9]+) (?:and )?¢([0-9]{1,2})\b", combine_cents, s)
        s = re.sub(r"[€£$]0.([0-9]{1,2})\b", extract_cents, s)

        # Write "one(s)" instead of "1(s)", just for readability
        s = re.sub(r"\b1(s?)\b", r"one\1", s)

        return s

    def __call__(self, s: str):
        s = self.preprocess(s)
        s = " ".join(word for word in self.process_words(s.split()) if word is not None)
        s = self.postprocess(s)

        return s


class EnglishSpellingNormalizer:
    """
    Applies British-American spelling mappings.
    Reference: https://www.tysto.com/uk-us-spelling-list.html
    """

    def __init__(self, english_spelling_mapping=english_spelling_normalizer):
        self.mapping = english_spelling_mapping

    def __call__(self, s: str):
        return " ".join(self.mapping.get(word, word) for word in s.split())


class EnglishTextNormalizer:
    """
    English text normalizer with:
    - Contraction expansion
    - Number normalization
    - British-American spelling conversion
    - Diacritic removal
    - Punctuation handling
    """

    def __init__(self, english_spelling_mapping=english_spelling_normalizer):
        self.ignore_patterns = r"\b(hmm|mm|mhm|mmm|uh|um)\b"
        self.replacers = {
            # Common contractions
            r"\bwon't\b": "will not",
            r"\bcan't\b": "can not",
            r"\blet's\b": "let us",
            r"\bain't\b": "aint",
            r"\by'all\b": "you all",
            r"\bwanna\b": "want to",
            r"\bgotta\b": "got to",
            r"\bgonna\b": "going to",
            r"\bi'ma\b": "i am going to",
            r"\bimma\b": "i am going to",
            r"\bwoulda\b": "would have",
            r"\bcoulda\b": "could have",
            r"\bshoulda\b": "should have",
            r"\bma'am\b": "madam",
            # Contractions in titles/prefixes
            r"\bmr\b": "mister ",
            r"\bmrs\b": "missus ",
            r"\bst\b": "saint ",
            r"\bdr\b": "doctor ",
            r"\bprof\b": "professor ",
            r"\bcapt\b": "captain ",
            r"\bgov\b": "governor ",
            r"\bald\b": "alderman ",
            r"\bgen\b": "general ",
            r"\bsen\b": "senator ",
            r"\brep\b": "representative ",
            r"\bpres\b": "president ",
            r"\brev\b": "reverend ",
            r"\bhon\b": "honorable ",
            r"\basst\b": "assistant ",
            r"\bassoc\b": "associate ",
            r"\blt\b": "lieutenant ",
            r"\bcol\b": "colonel ",
            r"\bjr\b": "junior ",
            r"\bsr\b": "senior ",
            r"\besq\b": "esquire ",
            # Perfect tenses
            r"'d been\b": " had been",
            r"'s been\b": " has been",
            r"'d gone\b": " had gone",
            r"'s gone\b": " has gone",
            r"'d done\b": " had done",
            r"'s got\b": " has got",
            # General contractions
            r"n't\b": " not",
            r"'re\b": " are",
            r"'s\b": " is",
            r"'d\b": " would",
            r"'ll\b": " will",
            r"'t\b": " not",
            r"'ve\b": " have",
            r"'m\b": " am",
        }
        self.standardize_numbers = EnglishNumberNormalizer()
        self.standardize_spellings = EnglishSpellingNormalizer(english_spelling_mapping)

    def __call__(self, s: str):
        # FIRST: Remove chat template role markers (before any other processing)
        # This ensures accurate WER/CER by excluding prompt tokens from comparison
        s = remove_chat_template_roles(s)

        s = s.lower()

        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # Remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # Remove words between parenthesis
        s = re.sub(self.ignore_patterns, "", s)
        s = re.sub(r"\s+'", "'", s)  # Standardize when there's a space before an apostrophe

        for pattern, replacement in self.replacers.items():
            s = re.sub(pattern, replacement, s)

        s = re.sub(r"(\d),(\d)", r"\1\2", s)  # Remove commas between digits
        s = re.sub(r"\.([^0-9]|$)", r" \1", s)  # Remove periods not followed by numbers
        s = remove_symbols_and_diacritics(s, keep=".%$¢€£")  # Keep some symbols for numerics

        s = self.standardize_numbers(s)
        s = self.standardize_spellings(s)

        # Now remove prefix/suffix symbols that are not preceded/followed by numbers
        s = re.sub(r"[.$¢€£]([^0-9])", r" \1", s)
        s = re.sub(r"([^0-9])%", r"\1 ", s)

        s = re.sub(r"\s+", " ", s)  # Replace any successive whitespace characters with a space

        return s
