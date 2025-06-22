# 1. Import libraries
import rospy
from std_msgs.msg import String

if __name__ == "__main__":
    # 2. Initialize the ROS node with a unique name
    rospy.init_node("talker_p")
    
    # 3. Create a Publisher object
    pub = rospy.Publisher("chatter", String, queue_size=10)
    
    # 4. Prepare the data to be published and implement the logic to publish it
    msg = String()
    msg_front = "hello "
    count = 0
    
    # Set loop frequency
    rate = rospy.Rate(1)
    
    while not rospy.is_shutdown():
        # Concatenate string
        msg.data = msg_front + str(count)
        
        pub.publish(msg)
        rate.sleep()
        rospy.loginfo("Published data: %s", msg.data)
        count += 1
