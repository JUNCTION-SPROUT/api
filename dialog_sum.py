import nlpcloud

client = nlpcloud.Client("bart-large-samsum", "", gpu=False, lang="en")
client.summarization("""Jules: Hey kids! How you boys doin’?
Jules: (Speaking to the guy laying on the couch) Hey, keep chillin’. You know who we are? We’re associates of your business partner Marsellus Wallace. You do remember your business partner don’t you? Let me take a wild guess here. You’re Brett, right?
Brett: Yeah.
Jules: I thought so. You remember your business partner Marsellus Wallace, don’t you, Brett?
Brett: Yeah, yeah, I remember him.
Jules: Good. Looks like me an Vincent caught you boys at breakfast. Sorry about that. Whatcha havin’?
Brett: Hamburgers.
Jules: Hamburgers! The cornerstone of any nutritious breakfast. What kind of hamburgers?
Brett: Ch-cheeseburgers.
Jules: No, no no, where’d you get ’em? McDonalds? Wendy’s? Jack in the Box? Where?
Brett: Big Kahuna Burger.
Jules: Big Kahuna Burger. That’s that Hawaiian burger joint. I hear they got some tasty burgers. I ain’t never had one myself. How are they?
Brett: They’re good.
Jules: Mind if I try one of yours? This is yours here, right?
Jules: (Picks up burger and takes a bite) Mmm-mmmm. That is a tasty burger. Vincent, ever have a Big Kahuna Burger?
(Vincent shakes his head)
Jules: Wanna bite? They’re real tasty.
Vincent: Ain’t hungry.""")