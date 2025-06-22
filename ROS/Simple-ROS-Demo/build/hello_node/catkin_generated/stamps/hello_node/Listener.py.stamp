# 1. Import libraries
import rospy
from std_msgs.msg import String

# Callback function to process received messages
def doMsg(msg):
    rospy.loginfo("I heard: %s", msg.data)

if __name__ == "__main__":
    # 2. Initialize the ROS node with a unique name
    rospy.init_node("listener_p")
    
    # 3. Create a Subscriber object
    sub = rospy.Subscriber("chatter", String, doMsg, queue_size=10)
    
    # 4. Handle subscribed messages (via the callback function)
    # 5. Continuously call the callback function in a loop
    rospy.spin()
