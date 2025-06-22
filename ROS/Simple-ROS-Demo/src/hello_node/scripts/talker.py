# 1. Import libraries
import rospy
from hello_node.msg import Person

if __name__ == "__main__":
    # 2. Initialize the ROS node with a unique name
    rospy.init_node("talker_person_p")
    # 3. Create a Publisher object
    pub = rospy.Publisher("chatter_person",Person,queue_size=10)
    # 4. Prepare the data to be published and implement the logic to publish it
    p = Person()
    p.name = "Qin"
    p.age = 22
    p.height = 1.75

    # Set loop frequency
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        pub.publish(p)
        rate.sleep() 
        rospy.loginfo("姓名:%s, 年龄:%d, 身高:%.2f",p.name, p.age, p.height)