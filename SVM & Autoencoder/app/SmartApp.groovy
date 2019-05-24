/**
 *  Abnormal Warning
 *
 *  Copyright 2018 YONGHUA YU
 *
 *  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 *  in compliance with the License. You may obtain a copy of the License at:
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
 *  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License
 *  for the specific language governing permissions and limitations under the License.
 *
 */
definition(
    name: "Abnormal Warning",
    namespace: "tcmch",
    author: "YONGHUA YU",
    description: "warn users about abnormal behavior in their home",
    category: "My Apps",
    iconUrl: "https://s3.amazonaws.com/smartapp-icons/Convenience/Cat-Convenience.png",
    iconX2Url: "https://s3.amazonaws.com/smartapp-icons/Convenience/Cat-Convenience@2x.png",
    iconX3Url: "https://s3.amazonaws.com/smartapp-icons/Convenience/Cat-Convenience@2x.png")


preferences {
    section("Choose Sensors"){
        input "motion", "capability.motionSensor", title: "Which Motion Sensor?"
        input "door", "capability.contactSensor", title: "Which Door Sensor?"
        input "acce", "capability.accelerationSensor", title: "Which Acceleration Sensor?"
        input "temp", "capability.temperatureMeasurement", title: "Which Temperature Sensor?"
    }
    section("Text me at..."){
        input "phone", "phone", title: "Phone number?", required: true
    }
}

def installed() {
    log.debug "Installed with settings: ${settings}"
    initialize()
}

def updated() {
    log.debug "Updated with settings: ${settings}"
    unsubscribe()
    initialize()
}

def initialize() {
    // TODO: subscribe to attributes, devices, locations, etc.
    subscribe(acce, "acceleration", eventHandler)
    subscribe(motion, "motion", eventHandler)
    subscribe(temp, "temperature", eventHandler)
    subscribe(door, "contact", eventHandler)
}

// TODO: implement event handlers
def eventHandler(evt) {
    log.trace "eventHandler($evt?.name: $evt?.value)"
    
    def cur_temp = temp.currentValue("temperature")
    def cur_motion = motion.currentValue("motion")
    def cur_acce = acce.currentValue("acceleration")
    def cur_door = door.currentValue("contact")
    /*log.debug "$cur_temp"
    log.debug "$cur_motion"
    log.debug "$cur_acce"
    log.debug "$cur_door"*/
    
    def now = new Date()
    log.debug "$now"
    
    /*def start_time
    use(groovy.time.TimeCategory){
        start_time = now-2.minutes
    }*/
    
    def params = [
    	uri: "http://35.243.207.59:5000/post",
        body: [
        	"motion": cur_motion,
            "acceleration": cur_acce,
            "door_status": cur_door,
            "temperature": cur_temp,
            "time": now
        ]
    ]
    try {
    	httpPostJson(params) {resp ->
        	def result = resp.data.data
        	log.debug "${result}"
            if(result == "-1")   // "-1" indicates abnormal, "1" indicate normal, "invalid input" indicates invalid input data  
            	sendSms(phone, "Abnormal behaviors detected in your home!!!")
        }
    } catch(e) {
    	log.error "something went wrong: $e"
    }
}
