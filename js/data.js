// Static data for the portfolio
const portfolioData = {
  projects: [
    {
      id: 1,
      title: "Weight Usage Analyzer",
      description: "ML toolkit for analyzing the effective use of weights in neural networks. It provides quantitative metrics (entropy, coverage, FLOPs) and visualizations to identify redundant parameters and guide model simplification, compatible with both PyTorch and TensorFlow.",
      image: "assets/images/weightanalyser.jpg",
      technologies: ["Python", "PyTorch", "TensorFlow", "Machine Learning", "Neural Networks", "Model Optimization"],
      github: "https://github.com/AngelLagr/WeightUsageAnalyzer",
      demo: "",
      featured: true,
      details: {
        overview: "Pruning and machine learning model compression techniques have been extensively studied. My goal was to investigate the importance of weights themselves, rather than just neurons or channels, to gain a finer understanding of model efficiency and complexity. This curiosity led to this project, an open-source Python library for detailed analysis of weight usage in neural networks (compatible with PyTorch & TensorFlow).<br><br>I also wanted to deepen my understanding of the mathematical foundations of weight importance analysis through specific metrics such as entropy:<br><div style='text-align: center; font-size: 1.1em; margin: 15px 0;'>H(W) = -∑ᵢ pᵢ log(pᵢ)</div>where pᵢ represents normalized weight magnitudes, enabling precise identification of redundant parameters.",
        objectives: [
          "Identify if models are oversized by finding layers or neurons that contribute little to the final decision",
          "Get concrete estimates of computational operations (FLOPs) required for training and inference",
          "Guide optimization decisions like reducing layer sizes without sacrificing accuracy",
          "Visualize the most important neural pathways to better understand model behavior",
          "Build models that maintain their power but with more efficiency and are lightweight",
        ],
        methodology: "Weight importance is calculated using a metric that combines weight magnitude with activation statistics.<br><br>Key components include:<br>• Entropy analysis using Shannon entropy to measure weight distribution uniformity<br>• FLOPs estimation for computational cost assessment<br>• Layer-wise efficiency reports with statistical measures<br><br>Architecture visualization employs graph theory principles where edge weights represent connection importance, helping me understand neural pathway efficiency.",
        challenges: "Ensuring mathematical rigor in weight importance calculations across different architectures, developing theoretically sound entropy measures for neural networks, and maintaining numerical stability in quantization schemes.<br><br>Additional challenges include balancing computational efficiency with analysis depth while preserving research validity across diverse network topologies.",
        futureWork: "Extension to transformer architectures with attention weight analysis, development of adaptive quantization schemes for improved model compression.<br><br>Future research directions include:<br>• Integration with neural architecture search (NAS) for automated optimization<br>• Real-time analysis capabilities for dynamic model optimization<br>• Theoretical framework extension to sparse attention mechanisms in large language models<br>• Hardware-aware pruning strategies for edge deployment"
      }
    },
    {
      id: 2,
      title: "Image Morphing Using Diffusers from Textual Descriptions",
      description: "Advanced research in text-to-image synthesis using diffusion models. Investigates prompt and image interpolation techniques to create smooth visual transitions, exploring the mathematical foundations of denoising diffusion probabilistic models (DDPMs) for controlled image generation.",
      image: "assets/images/exemple.gif",
      technologies: ["Python", "Diffusers", "Stable Diffusion", "DDPM", "Computer Vision", "Text-to-Image", "Research"],
      github: "https://github.com/AngelLagr/Image-Morphing-using-Diffusers-from-Textual-Descriptions",
      demo: "",
      featured: true,
      details: {
        overview: "This project explores the mathematical foundations and practical applications of diffusion models for text-conditioned image synthesis and interpolation.<br><br>I wanted to investigate denoising diffusion probabilistic models (DDPMs) following the core formulation:<br><br><div style='text-align: center; font-size: 1.1em; margin: 15px 0;'>q(xₜ|xₜ₋₁) = N(xₜ; √(1-βₜ)xₜ₋₁, βₜI)</div><br>where βₜ represents the noise schedule.<br><br> I'm using pre-trained models, the goal more about understanding these models and their hyperparameter rather that coding them from scratch.",
        objectives: [
          "Investigate the mathematical foundations of denoising diffusion probabilistic models",
          "Understand the role of hyperparameters in diffusion processes",
          "Implement and analyze prompt engineering techniques for enhanced visual narratives"
        ],
        methodology: "I used diffusers library, and create a simple pipeline for text-to-image morphing.",
        results: "I succeded to obtain results like this one : <div style='text-align: center; margin: 15px 0;'><img src='assets/images/exemple.gif' alt='Example Result'></div>"
      }
    },
    {
      id: 3,
      title: "Deep Q-Learning Space Navigator",
      description: "Reinforcement learning project to optimize space navigation in a simulated solar system. Trains a Deep Q-Learning agent to pilot a rocket from Earth to target planets while minimizing energy consumption and travel time.",
      image: "assets/images/space.jpg",
      technologies: ["Python", "PyTorch", "Reinforcement Learning", "Deep Q-Learning", "Pygame", "Simulation"],
      github: "https://github.com/AngelLagr/Deep-Q-Learning-Space-Navigator",
      demo: "",
      featured: true,
      details: {
        overview: "This project explores the use of Deep Q-Learning to optimize space trajectories in a simulated solar system. My goal was to train an agent to launch from Earth and reach a target planet (e.g., Mars) using reinforcement learning.<br><br>I used Pygame for visualization, PyTorch for deep learning, and built a physics simulation that accounts for planetary gravity. The current version works with a simple environment, but my ultimate ambition is to generalize the approach to compute optimal trajectories in any dynamic solar system configuration.<br><br>I'm working on this more accessible version of the problem to fully grasp its complexities and tackle it step by step. The idea is to eventually extend the algorithm to a dynamic system, enabling automated space missions to be planned in any planetary configuration.",
        objectives: [
          "Train a Deep Q-Learning agent for space trajectory optimization",
          "Minimize energy consumption and travel time from Earth to Mars",
          "Handle complex gravitational environments with multiple celestial bodies",
          "Develop autonomous space navigation capabilities",
          "Extend from static to dynamic planetary configurations",
          "Create a foundation for automated space mission planning"
        ],
        methodology: "I implemented Deep Q-Learning algorithms in a realistic space simulation environment with accurate gravitational physics. The agent learns to make optimal navigation decisions by balancing fuel efficiency with mission constraints.<br><br>Key technical components:<br>• Pygame for real-time visualization of trajectories<br>• PyTorch for the Deep Q-Learning implementation<br>• Custom physics simulation accounting for planetary gravity<br>• Reward function balancing fuel consumption and mission success<br><br>The current approach focuses on static planetary positions to understand the core mechanics before tackling the full dynamic problem.",
        results: "The agent successfully learns to navigate from Earth to Mars with optimized trajectories. Training results show clear improvement over episodes, with rewards increasing as the model learns more efficient paths.<br><br>Here's an example of the result of a whole training process, we can see that the model is more and more rewarded throughout the training :<br><br><div style='text-align: center; margin: 15px 0;'><img src='assets/images/performancespace.png' alt='Example Result' style='width: 60%; max-width: 500px;'></div>",
        challenges: "I really struggled to make the model learn effectively and converge, especially in the early stages where it had to explore a vast action space. Balancing exploration and exploitation was crucial, and I had to fine-tune hyperparameters like learning rate, discount factor, and exploration strategy.",
        futureWork: "The next major step is extending to dynamic planetary configurations where celestial bodies move according to real orbital mechanics.<br><br>Future developments include:<br>• Integration with real orbital mechanics software<br>• Multi-target mission planning capabilities<br>• Real-time trajectory adaptation for changing conditions <br>• Exploration of advanced RL techniques like Proximal Policy Optimization (PPO)"
      }
    },
    {
      id: 4,
      title: "Convolutional Autoencoder for Image Compression",
      description: "Explores image compression using convolutional autoencoders. Focuses on compressing satellite images from the EuroSAT dataset into compact representations and reconstructing them with minimal quality loss.",
      image: "assets/images/flou.webp",
      technologies: ["Python", "PyTorch", "Machine Learning", "Convolutional Neural Networks", "Computer Vision", "Image Processing", "EuroSAT", "Matplotlib"],
      github: "https://github.com/AngelLagr/Convolutional-Autoencoder-for-Image-Compression",
      demo: "",
      featured: false,
      details: {
        overview: "This project focuses on image compression using a convolutional autoencoder. My goal was to compress satellite images from the EuroSAT dataset into compact representations and reconstruct them with minimal quality loss.<br><br>I wanted to explore how autoencoders could learn efficient image representations for compression tasks. The model uses PyTorch and is trained to minimize the difference between original and reconstructed images using mean squared error (MSE) loss with the Adam optimizer.<br><br>I worked with EuroSAT satellite images, resizing them to 64x64 pixels during training to make the process more manageable while still capturing the essential features of satellite imagery.",
        objectives: [
          "Develop efficient image compression techniques using autoencoders",
          "Minimize quality loss during the compression-reconstruction process",
          "Learn compact representations of satellite imagery characteristics",
          "Implement and train a convolutional autoencoder from scratch",
          "Evaluate reconstruction quality on satellite images"
        ],
        methodology: "I implemented a convolutional autoencoder architecture specifically designed for satellite imagery compression. The model consists of an encoder that compresses images into low-dimensional latent representations and a decoder that reconstructs the original images.<br><br>Key technical components:<br>• PyTorch implementation of the autoencoder architecture<br>• Mean Squared Error (MSE) loss function for training<br>• Adam optimizer for efficient gradient descent<br>• EuroSAT dataset preprocessing and 64x64 pixel resizing<br>• Evaluation metrics focused on reconstruction quality",
        results: "I successfully trained the autoencoder to compress and reconstruct satellite images with good visual quality. The model learned to capture the essential features of satellite imagery while significantly reducing file size.<br><br>The project includes comprehensive testing capabilities with both dataset images and larger custom images to evaluate compression performance across different scales.",
        challenges: "I faced several challenges during this project, particularly in balancing compression ratio with reconstruction quality. Finding the right latent space dimensionality was crucial . Too small and important details were lost, too large and compression wasn't effective.",
        futureWork: "I'd like to extend this work to handle real-time compression for satellite communications and explore more advanced architectures like variational autoencoders (VAEs).<br><br>Future improvements could include:<br>• Integration with edge computing platforms for on-satellite processing<br>• Adaptation to other specialized imagery domains beyond satellites<br>• Implementation of perceptual loss functions for better visual quality<br>• Exploration of attention mechanisms for better feature preservation"
      }
    },
    {
      id: 8,
      title: "Active Learning with Uncertainty Sampling",
      description: "Explores active learning techniques to classify MNIST digits with minimal training data. Uses uncertainty sampling to iteratively select the most informative examples, achieving better performance with fewer labeled samples.",
      image: "assets/images/learning.jpg",
      technologies: ["Python", "TensorFlow", "Keras", "Scikit-learn", "Active Learning", "CNN", "Matplotlib", "NumPy"],
      github: "https://github.com/AngelLagr/Active-Learning-with-Uncertainty-Sampling",
      demo: "",
      featured: false,
      details: {
        overview: "I wanted to explore how Active Learning can help a model improve faster by selecting the most uncertain examples to label and learn from, rather than just using more data.<br><br>The goal was to classify MNIST digits but with much less data during the training process. I used a simple CNN, started with just 100 labeled samples, and then iteratively added the most 'confusing' images using a margin sampling strategy.<br><br>During each epoch, I predict all instances and select for the next epoch those with the lowest predicted class probabilities. In parallel, I trained a standard baseline model that just gets random new samples at each iteration to compare if this 'smart training' is actually worth it.",
        objectives: [
          "Classify MNIST digits with minimal training data",
          "Implement uncertainty sampling for active learning",
          "Compare active learning vs random sampling strategies",
          "Demonstrate efficiency of smart data selection",
          "Achieve better performance with fewer labeled samples"
        ],
        methodology: "I implemented an active learning framework using uncertainty sampling with margin-based selection strategy.<br><br>Key technical components:<br>• Simple CNN for MNIST digit classification<br>• Margin sampling strategy for uncertainty estimation<br>• Iterative training with selective sample addition<br>• Baseline comparison with random sampling<br>• Performance tracking across learning iterations",
        results: "After 5 rounds of active learning using only ~150 samples total, the results showed clear benefits of the uncertainty sampling approach.<br><br>The active learner improved faster than the baseline, especially in early rounds, demonstrating that the model benefits from choosing the right training points rather than just adding more data.<br><br>Here's the learning curve comparison showing active learning vs random sampling:<br><br><div style='text-align: center; margin: 15px 0;'><img src='assets/images/exemplelearning.png' alt='Active Learning Results' style='width: 70%; max-width: 500px;'></div>",
        futureWork: "I'd like to explore more sophisticated uncertainty estimation methods and apply this to more complex datasets.<br><br>Future improvements could include:<br>• Testing with more advanced uncertainty measures like entropy or disagreement<br>• Applying to real-world datasets beyond MNIST<br>• Implementing ensemble-based uncertainty estimation<br>• Exploring cost-sensitive active learning strategies"
      }
    },
    {
      id: 9,
      title: "Recreating a Simplified YOLO Model",
      description: "A simplified recreation of the YOLO (You Only Look Once) object detection model developed as part of engineering coursework. Focuses on identifying and localizing objects within images using a lightweight YOLO implementation.",
      image: "assets/images/rino.jpg",
      technologies: ["Python", "Computer Vision", "Object Detection", "YOLO", "Deep Learning", "CNN"],
      github: "https://github.com/AngelLagr/Recreating-a-Simplified-YOLO-Model",
      demo: "",
      featured: false,
      details: {
        overview: "This project is a simplified recreation of the YOLO (You Only Look Once) object detection model. Developed as part of a course in my Engineering school (ENSEEIHT), it aims to identify and localize objects within an image using a lightweight version of YOLO.<br><br>The goal was to understand the fundamental concepts behind real-time object detection and implement a working version from scratch. This was both a learning exercise and a practical implementation of one of the most influential object detection architectures.",
        objectives: [
          "Understand and implement YOLO architecture concepts",
          "Create a lightweight object detection model",
          "Learn real-time object detection techniques",
          "Apply computer vision knowledge from coursework",
          "Build a working object localization system"
        ],
        methodology: "I implemented a simplified version of the YOLO architecture, focusing on the core concepts of single-shot object detection.<br><br>Key technical components:<br>• Simplified YOLO network architecture<br>• Bounding box regression and classification<br>• Grid-based object detection approach<br>• Loss function implementation for detection tasks<br>• Training pipeline for object detection"
      }
    },
    {
      id: 5,
      title: "ML Asteroid Impact Prediction",
      description: "Uses machine learning to assess the danger of near-Earth objects (NEOs). Built a Random Forest Classifier that analyzes orbital features like size, eccentricity, and orbital period to classify asteroids as dangerous or non-dangerous.",
      image: "assets/images/meteore.webp",
      technologies: ["Python", "Machine Learning", "Scikit-learn", "Random Forest", "SMOTE", "Data Analysis", "Classification" , "Pandas"],
      github: "https://github.com/AngelLagr/ML-Asteroid-Impact-Prediction",
      demo: "",
      featured: false,
      details: {
        overview: "When I heard about asteroid 2024 YR4 getting attention with a reported 3% chance of colliding with Earth, I decided to explore whether machine learning could help assess the danger of near-Earth objects.<br><br>My goal was to create a model that classifies asteroids based on their orbital parameters (diameter, eccentricity, etc.) to determine if they pose a threat to our planet. I used a Random Forest Classifier, but since the data wasn't balanced, I applied SMOTE and RandomUnderSampler to handle the imbalanced dataset properly.",
        objectives: [
          "Create a model to classify asteroids as dangerous or non-dangerous",
          "Use orbital parameters like diameter and eccentricity for prediction",
          "Handle imbalanced datasets with proper sampling techniques",
          "Test the model on real asteroids including 2024 YR4"
        ],
        methodology: "I implemented a Random Forest Classifier to analyze multiple orbital parameters of near-Earth objects. To handle the imbalanced dataset, I used SMOTE for minority class augmentation and RandomUnderSampler for balanced training.<br><br>Key technical components:<br>• Random Forest Classifier for robust prediction<br>• SMOTE and RandomUnderSampler for dataset balancing<br>• Kaggle asteroid dataset for training and validation<br>• Python libraries: scikit-learn, imblearn, pandas, joblib",
        results: "I achieved strong performance metrics with 99.99% accuracy, 97.66% precision, 98.43% recall, and 98.04% F1 score. The confusion matrix showed [[59870, 3], [2, 125]].<br><br>When tested on real asteroids, including a known dangerous one from the past, it correctly flagged it as dangerous. Fun part : For 2024 YR4, the model predicted it as non-dangerous, phew ! <br><br><strong>Note:</strong> As an engineering student, this work represents an educational exploration rather than professional astronomical assessment !",
        challenges: "The main challenge was dealing with the heavily imbalanced dataset since dangerous asteroids are rare. I had to carefully implement sampling techniques to ensure the model could learn from both classes effectively.<br><br>I also needed to be cautious about overfitting, especially given the very high accuracy scores.",
      }
    },
    {
      id: 6,
      title: "SRGAN for Image Upscaling",
      description: "Built my first Super-Resolution Generative Adversarial Network (SRGAN) to learn about adversarial training and image upscaling. Focused on understanding GAN concepts and achieving working results with minimal training epochs.",
      image: "assets/images/upscale.jpg",
      technologies: ["Python", "GANs", "Deep Learning", "Computer Vision", "Image Enhancement", "Super-Resolution", "SRGAN"],
      github: "https://github.com/AngelLagr/GAN-for-Upscaling-Pictures",
      demo: "",
      featured: false,
      details: {
        overview: "My goal was to build an SRGAN capable of upscaling the resolution of pictures. This was my first experience working with GANs, and I wanted to understand the concept of Super-Resolution GANs and make them work with Python.<br><br>I used resources from python_for_microscopists GitHub repository along with knowledge from my engineering courses to implement this project. The focus was on learning the fundamentals of adversarial training and super-resolution techniques.",
        objectives: [
          "Understand the concept and implementation of SRGANs",
          "Build a working super-resolution model from scratch",
          "Learn adversarial training techniques",
          "Achieve meaningful image upscaling results",
          "Gain hands-on experience with GANs"
        ],
        methodology: "I implemented a Super-Resolution Generative Adversarial Network using Python, focusing on understanding the core concepts rather than achieving state-of-the-art results.<br><br>Key technical components:<br>• Generator network for image super-resolution<br>• Discriminator network for adversarial training<br>• Perceptual loss functions for better visual quality<br>• Training pipeline with careful loss balancing<br>• Python implementation using deep learning frameworks",
        results: "I achieved working results with only 10 epochs of training. The model successfully upscaled low-resolution images to higher resolutions with noticeable improvements. I could go further but it was not my goal to try to have the best model possible. <br><br>Here's an example result (left: low resolution input, middle: SRGAN output, right: high resolution target):<br><br><div style='text-align: center; margin: 15px 0;'><img src='assets/images/srgan_results.png' alt='SRGAN Results' style='width: 80%; max-width: 600px;'></div>",
        challenges: "I had significant difficulty making the GAN converge properly. Balancing the generator and discriminator training was particularly challenging, and I had to carefully tune the learning rates and loss functions to achieve stable training.<br><br>The adversarial training process was quite sensitive to hyperparameters, and it took many attempts to get meaningful results.",
        futureWork: "I'd like to experiment with more stable GAN architectures for better convergence and results.<br><br>Future improvements could include:<br>• Testing with more stable models like IDSGAN or similar architectures<br>• Implementing progressive training techniques<br>• Exploring different loss functions for better perceptual quality<br>"
      }
    },
    {
      id: 7,
      title: "CNN Model for Facial Emotion Recognition",
      description: "Built a Convolutional Neural Network to recognize emotions on faces. Applied machine learning concepts to computer vision for emotion detection and classification across multiple emotional states.",
      image: "assets/images/yolo.gif",
      technologies: ["Python", "CNNs", "Computer Vision", "Deep Learning", "Emotion Analysis"],
      github: "https://github.com/AngelLagr/Emotion-recognize-with-AI",
      demo: "",
      featured: false,
      details: {
        overview: "The aim of the project is to use what I've learned in class to apply it to another field! So I'm trying to build a model capable of recognizing emotions on a face using convolutional neural networks.",
        objectives: [
          "Apply CNN techniques to emotion recognition",
          "Classify multiple emotional states",
          "Develop practical emotion detection system",
          "Bridge academic learning with real applications"
        ],
        methodology: "Uses convolutional neural network architectures optimized for facial emotion recognition. The model processes facial features and expressions to classify emotions across multiple categories.",
        results: "Successfully recognizes various emotional states from facial expressions, demonstrating practical application of deep learning in computer vision.",
      }
    },
    {
      id: 10,
      title: "Connect Four AI Competition",
      description: "Built a Python Connect Four AI for a class competition. Created an intelligent algorithm to compete against other students' implementations and win the tournament.",
      image: "assets/images/c4.webp",
      technologies: ["Python", "Artificial Intelligence", "Game Theory", "Algorithms", "Minimax"],
      github: "https://github.com/AngelLagr/Create-a-perfect-opponent-for-Connect-4",
      demo: "",
      featured: false,
      details: {
        overview: "My Python code for the 'Puissance4 IA' project. The goal was to create a class competition: build the best Connect Four script and beat everyone! This was a fun competitive programming challenge during my studies.",
        objectives: [
          "Create the best Connect Four AI for class competition",
          "Implement game theory algorithms",
          "Beat other students' algorithms",
          "Learn competitive programming"
        ],
        methodology: "I implemented a minimax algorithm with heuristic evaluation to create a competitive Connect Four player for the class tournament.",
        results: "Created a working Connect Four AI that competed in the class tournament.",
        challenges: "Balancing computational efficiency with playing strength within the competition constraints.",
        futureWork: "Plans to enhance with reinforcement learning techniques for better performance."
      }
    }
  ],

  techStack: [
    { name: "Python", icon: "assets/icons/python-svgrepo-com.svg" },
    { name: "TensorFlow", icon: "assets/icons/tensorflow-svgrepo-com.svg" },
    { name: "Keras", icon: "assets/icons/Keras.svg" },
    { name: "PyTorch", icon: "assets/icons/pytorch-svgrepo-com.svg" },
    { name: "Scikit-Learn", icon: "assets/icons/Scikit_learn_logo_small.svg" },
    { name: "OpenCV", icon: "assets/icons/opencv-svgrepo-com.svg" },
    { name: "OpenMP", icon: "assets/icons/OpenMP_logo.svg" },
    { name: "Hugging Face", icon: "assets/icons/huggingface-color.png" },
    { name: "NumPy", icon: "assets/icons/numpy-svgrepo-com.svg" },
    { name: "Pandas", icon: "assets/icons/pandas.png" },
    { name: "Matplotlib", icon: "assets/icons/matplotlib.png" },
    { name: "SciPy", icon: "assets/icons/SCIPY_2.svg" },
    { name: "GitHub", icon: "assets/icons/github-color-svgrepo-com.svg" },
  ],

  blogPosts: [
    {
      id: 8,
      title: "Running Large Language Models Locally with languagemodels",
      excerpt: "A practical guide to using the languagemodels Python library for local LLM inference with minimal memory requirements...",
      content: `
        <p>Running large language models locally can be challenging due to memory constraints and complex setup processes. The <code>languagemodels</code> Python library by <a href="https://github.com/jncraton/languagemodels" target="_blank">jncraton</a> simplifies this by enabling LLM inference in as little as 512MB of RAM.</p>
        
        <h3>Installation</h3>
        <p>Getting started is straightforward with pip:</p>
        <pre><code>pip install languagemodels</code></pre>
        
        <h3>Basic Usage</h3>
        <p>The library provides a simple interface for common tasks:</p>
        <pre><code>import languagemodels as lm

lm.do("What color is the sky?")
# Output: 'The color of the sky is blue.'
</code></pre>
        
        <p>The first run requires downloading model data (~250MB), but subsequent calls are fast thanks to local caching.</p>
        
        <h3>Memory Configuration</h3>
        <p>You can adjust memory usage to access more powerful models:</p>
        <pre><code># Default model (512MB)
lm.do("If I have 7 apples then eat 5, how many do I have?")
# Output: 'You have 8 apples.' (incorrect)

# Increase memory for better performance
lm.config["max_ram"] = "4gb"
lm.do("If I have 7 apples then eat 5, how many do I have?")
# Output: 'I have 2 apples left.' (correct)</code></pre>
        
        This lib have also other features like completion, etc..
        That's why I found it really useful for various tasks and by keeping the memory usage low, it can be used on many devices, even Raspberry Pi !

        <h3>Document Storage and Semantic Search</h3>
        <p>You can store documents and perform semantic search for context retrieval:</p>
        <pre><code># Store documents
lm.store_doc(lm.get_wiki("Python"), "Python")
# Search for relevant context
context = lm.get_doc_context("What is a int in Python?")
# Returns relevant passages from stored documents</code></pre>
        
        <h3>Performance</h3>
        <p>The library uses int8 quantization and CTranslate2 backend for efficient CPU inference:</p>
        <ul>
          <li>2x faster than Hugging Face transformers</li>
          <li>5x less memory usage</li>
          <li>Negligible quality loss from quantization</li>
        </ul>

        <h3>Practical Applications</h3>
        <p>I found it really useful for:</p>
        <ul>
          <li>Learning about language models with hands-on experimentation</li>
          <li>Building privacy-focused applications with local inference</li>
          <li>Resource-constrained environments where efficiency matters</li>
        </ul>
        
        <p>The simplicity of the API makes it an accessible entry point for working with language models locally, though performance will be below current state-of-the-art cloud models.</p>

        <h3>Conclusion</h3>
        <p>The <code>languagemodels</code> library can really be useful for studying and experimenting with language models locally. That's why I recommend people that want to understand how RAG works or how llm interacts with user queries to give it a try.</p>

        <h3>References</h3>
        <p>🔗 <strong>Project Repository:</strong> <a href="https://github.com/jncraton/languagemodels" target="_blank">https://github.com/jncraton/languagemodels</a><br>
        📝 <strong>Author:</strong> jncraton</p>
      `,
      image: "assets/images/llm_local.png",
      date: "2025-07-15",
      readTime: "5 min read",
      tags: ["Python", "LLM", "Local Inference", "Machine Learning", "NLP", "Tutorial"],
      type: "tutorial"
    },
    {
      id: 1,
      title: "Reducing Machine Learning Costs: A Study on Model Optimization Techniques",
      excerpt: "⚡ Quick report on a study I conducted on reducing Machine Learning costs through pruning and quantization...",
      content: `
        <p>⚡ Quick report on a study I conducted on reducing Machine Learning costs ⚡</p>
        
        <p>I believe that reducing the energy impact (and therefore cost) of AI models is becoming increasingly crucial given the place these models occupy today, and several solutions exist for this.</p>
        
        <h3>The Techniques Explored</h3>
        <p>I explored two ML model optimization techniques presented by a publication that was recommended to me (<a href="https://arxiv.org/abs/2307.02973" target="_blank">research paper</a>):</p>
        <ul>
          <li><strong>Pruning</strong> ➡️ We remove "useless" connections and neurons in a neural network</li>
          <li><strong>Quantization</strong> ➡️ We transform the numerical values that models use for their operations into a much lighter format to process (float32 → int8 for the curious)</li>
        </ul>
        
        <h3>My Implementation</h3>
        <p>I applied these techniques to a small Keras model trained on the "Wine" dataset (💪🍷) using my personal weight usage analysis tool (<a href="https://github.com/AngelLagr/weight-usage-analyzer" target="_blank">WeightUsageAnalyzer</a>) and TensorFlow's optimization tools. My goal was to measure their energy and economic impact before and after optimization.</p>
        
        <h3>Results 👇</h3>
        <p><strong>Baseline model:</strong><br>
        ➡️ The base model performs about 320k operations per use.<br>
        ➡️ Running it 30 million times per second (like Amazon Search might do), for a year, would consume ~1,120,397 kWh, or about €324,915 in electricity.</p>
        
        <p><strong>With pruning ✂️:</strong><br>
        With the "pruned" version of the model (same performance, fewer weights, fewer neurons) we drop to: ~€217,840 in electricity per year for the same accuracy, saving ~€107,075 / year, just by optimizing the architecture!</p>
        
        <p><strong>With quantization 🧮:</strong><br>
        Quantization reduces the calculations needed for a model operation, thus its unit energy cost. An int8 calculation can consume up to <strong>18.5× less energy</strong> than float32. We almost divide the model's energy cost by 20!</p>
        
        <h3>Conclusion 💰</h3>
        <p>➡️ So, by converting the "pruned" model to int8, we drop to ~€11,775 / year versus ~€325k original energy cost! That's <strong>96.4% savings</strong>.</p>
        
        <p>For those who want to see how I obtained all these results, everything is explained in a Notebook on my GitHub: <a href="https://github.com/AngelLagr/reduce-ml-cost-with-quantization-pruning" target="_blank">GitHub Repository</a></p>
        
        <blockquote>
          "Obviously, my little wine classification model won't be called billions of times like Amazon's. But most AI models in production are much larger so their impact is even greater. AI is a polluting technology and it's up to us, its creators and users, to do our best to improve this!"
        </blockquote>
        
        <p><strong>PS:</strong> I use sources here that may have evolved over time, and the calculations presented in this notebook remain approximate. The objective of this demo is not absolute precision, but rather to show how important it is to think about how we build our ML models, especially when we scale them up.</p>
      `,
      image: "assets/images/energy.jpg",
      date: "2025-07-20",
      readTime: "6 min read",
      tags: ["Machine Learning", "Optimization", "Energy Efficiency", "Cost Reduction", "Pruning", "Quantization"],
      type: "research"
    },
    {
      id: 2,
      title: "Project Presentation: Weights, Efficiency, and Computational Sobriety in Machine Learning🚦",
      excerpt: "Exploring weight-level analysis in neural networks to understand individual parameter contributions and optimize model efficiency...",
      content: `
        <p>Project presentation: Weights, Efficiency, and Computational Sobriety in Machine Learning 🚦</p>
        
        <p>We often aim to use the most powerful models, but it's also important to avoid using unnecessarily large models for simple problems. This kind of optimization is crucial to reducing, for exemple, the energy impact of machine learning.</p>
        
        <h3>Context</h3>
        <p>Pruning and model compression techniques are things that have been extensively studied by researchers. But during my personal research, I noticed that most pruning techniques focus on analyzing neurons rather than the weights themselves. I wanted to explore this less-studied angle to better understand the actual contribution of individual weights to the model's efficiency and complexity.</p>
        
        <h3>My tool</h3>
        <p>This curiosity led to this project, an open-source Python toolkit for detailed analysis of weight usage in neural networks (compatible with PyTorch & TensorFlow).</p>
        
        <p>In many architectures, a large portion of weights contributes very little to the network's output. Studying weight importance allows to:</p>
        <ul>
          <li>Assess overparameterization beyond neuron-level metrics</li>
          <li>More accurately estimate true computational cost (FLOPs) during training and inference</li>
          <li>Identify precise opportunities for model size reduction without loss of performance</li>
          <li>Visualize actual information flow paths in the network</li>
        </ul>
        
        <h3>Conclusion</h3>
        <p>The tool I built helps gain access to:</p>
        <ul>
          <li>Objective metrics on weight importance (entropy, top-k coverage, etc.)</li>
          <li>Detailed layer-wise reports</li>
          <li>Automatic FLOPs estimation</li>
          <li>Visualizations where connection thickness and color reflect actual utility</li>
        </ul>
        
        <h3>Visualization Example</h3>
        <p>You can find an example of the possible visualization here:</p>
        
        <div style='text-align: center; margin: 20px 0;'>
          <img src='assets/images/usage.jpg' alt='WeightUsageAnalyzer Visualization' style='width: 80%; max-width: 600px; border-radius: 8px;'>
        </div>
        
        <h3>Open Source & Available Now</h3>
        <p>All code, documentation, and examples are available here:<br>
        🔗 <a href="https://github.com/AngelLagr/weight-usage-analyzer" target="_blank">https://github.com/AngelLagr/weight-usage-analyzer</a></p>
        
        <p>Feel free to try it and if you find it useful, a star is always appreciated!</p>
        
      `,
      image: "assets/images/weightanalyser.jpg",
      date: "2025-06-15",
      readTime: "4 min read",
      tags: ["Deep Learning", "AI", "Open Source", "Model Compression", "Computational Efficiency", "Weight Analysis"],
      type: "project"
    },
    {
      id: 4,
      title: "Experimenting on the NASA’s Asteroid Dataset with Machine Learning🪐",
      excerpt: "Using machine learning to assess the danger of near-Earth objects, including the famous asteroid 2024 YR4...",
      content: `
        <p>🪐 Small presentation of a project I've been working on recently! 🪐</p>
        
        <p>As you probably know, space threats are causing a lot of concern, particularly the famous asteroid 2024 YR4, which recently reached a 3% probability of hitting Earth! So I wondered if it was possible to use machine learning to assess the danger of this asteroid and other near-Earth objects (NEOs).</p>
        
        <h3>The Project</h3>
        <p>I developed a machine learning model (trained with ~140,000 data points) based on a Random Forest Classifier to analyze different characteristics of asteroids, such as their diameter, orbital eccentricity, and other parameters. The idea is to determine whether an asteroid represents a potential danger to our planet.</p>
        
        <h3>Results 💡</h3>
        <p>After training my model, here's what I obtained with my validation data (~60,000 data points):</p>
        
        <ul>
          <li><strong>Model Accuracy:</strong> 99.70%</li>
          <li><strong>Precision:</strong> 97.66%</li>
          <li><strong>Recall:</strong> 98.43%</li>
          <li><strong>F1 Score:</strong> 98.04%</li>
        </ul>
        
        <p>And a rather convincing confusion matrix:</p>
        <div style="text-align: center;">
[59870&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3] <br>
[&nbsp;&nbsp;&nbsp;&nbsp;2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;125]
        </div>
        <p>Only 5 samples were incorrectly predicted by the model out of all the data!</p>
        
        <h3>How It Works 📊</h3>
        <p>I used SMOTE to balance the classes, given the rarity of dangerous asteroids in the dataset. I then applied a RandomUnderSampler to optimally adjust the sample. This way, we get a training dataset that contains as much information about dangerous asteroids as about non-dangerous asteroids.</p>
        
        <p>Finally, I trained the model with a Random Forest to predict asteroid danger based on their orbital characteristics.</p>
        
        <h3>The Cherry on Top 🍒</h3>
        <p>And the cherry on top: when I tested my model on 2024 YR4, it was classified as non-dangerous. No collision on the horizon... but we can never be too careful! 😅</p>
        
        <p>Of course, this model is just an approximation and shouldn't be taken as gospel... I'm just a student playing with AI, not an astrophysics expert! For a definitive assessment of the situation, we'll probably have to wait a bit longer...</p>
        
        <p><strong>🔗 Source Code:</strong> You can learn more and check out the source code on my GitHub repository: <a href="https://github.com/AngelLagr/ML-Asteroid-Impact-Prediction" target="_blank">ML Asteroid Impact Prediction</a></p>
      `,
      image: "assets/images/meteore.webp",
      date: "2025-02-10",
      readTime: "5 min read",
      tags: ["Machine Learning", "Asteroid Prediction", "Random Forest", "Space", "Data Science"],
      type: "research"
    },
    {
      id: 3,
      title: "My Researches in Reinforcement Learning: Space Navigation",
      excerpt: "Building a Deep Q-Learning agent to pilot spacecraft in complex gravitational environments...",
      content: `
        <p>Space navigation has always fascinated me, combining the beauty of physics with the power of artificial intelligence. In this post, I'll present my little project of building a reinforcement learning agent for spacecraft trajectory optimization.</p>
        
        <h3>The Inspiration</h3>
        <p>The idea came to me while watching a Ariane launch. I wondered: could AI learn to navigate space as efficiently as human mission planners, but faster and with less fuel consumption? AI can find optimal trajectories that we would never think of, why not use it for spacecraft navigation?</p>

        <h3>Technical Implementation</h3>
        <p>I used Deep Q-Learning with experience replay, training the agent in a simulated solar system with realistic orbital mechanics. The environment included:</p>
        <ul>
          <li>Multiple gravitational bodies</li>
          <li>Fuel consumption constraints</li>
          <li>Time-optimal trajectory planning</li>
          <li>Dynamic obstacle avoidance</li>
        </ul>
        
        <h3>Challenges Faced</h3>
        <p>The main challenges were dealing with sparse rewards and the continuous nature of space. I solved this by implementing a custom reward shaping function that balanced fuel efficiency with mission time and by picking more frequently the experience with the best reward for the training batch. I also had difficulties making the model converge.</p>

        <h3>Results</h3>
        <p>After a lot of episodes of training, the agent achieved to complete his mission with a good trajectory that minimized fuel consumption while ensuring timely arrival at the destination. However, for now, the model is acting in a simplified environment and may not generalize well to more complex scenarios. But I'm optimistic about pushing it to more various environment and try it to adapt to real-world conditions!</p>
      `,
      image: "assets/images/fusee.PNG",
      date: "2025-01-15",
      readTime: "2 min read",
      tags: ["Reinforcement Learning", "Space", "Deep Q-Learning", "Simulation"],
      type: "research"
    },
    {
      id: 5,
      title: "Attention Is All You Need - Transformer Architecture",
      excerpt: "Foundational paper introducing the Transformer architecture that revolutionized NLP and became the backbone of modern language models...",
      content: `
        <h3>Paper Overview</h3>
        <p><strong>Authors:</strong> Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin (Google Brain/Research)</p>
        <p><strong>Published:</strong> NIPS 2017</p>
        <p><strong>arXiv:</strong> <a href="https://arxiv.org/abs/1706.03762" target="_blank">1706.03762</a></p>
        
        <h3>Key Contributions</h3>
        <p>This groundbreaking paper introduced the Transformer architecture, which relies entirely on attention mechanisms without recurrence or convolution. The key innovations include:</p>
        <ul>
          <li><strong>Self-Attention Mechanism:</strong> Allows the model to weigh the importance of different parts of the input sequence</li>
          <li><strong>Multi-Head Attention:</strong> Enables the model to attend to different representation subspaces simultaneously</li>
          <li><strong>Positional Encoding:</strong> Injects sequence order information without recurrence</li>
          <li><strong>Parallelization:</strong> Significantly faster training compared to RNNs</li>
        </ul>
        
        <h3>Personal Notes & Analysis</h3>
        <p>What struck me most about this paper is how the authors completely departed from the sequential processing paradigm that dominated NLP. The attention mechanism allows for direct connections between any two positions in the sequence, enabling better capture of long-range dependencies.</p>
        
        <p>The mathematical elegance of the scaled dot-product attention is particularly noteworthy:</p>
        <p><code>Attention(Q,K,V) = softmax(QK^T/√d_k)V</code></p>
        
        <h3>Impact on My Research</h3>
        <p>This paper fundamentally changed how I think about sequence modeling. The principles of attention have applications far beyond NLP - I've been exploring how similar mechanisms could be applied to my weight analysis work in neural networks.</p>
        
        <h3>Key Takeaways</h3>
        <ul>
          <li>Sometimes the best solution is to completely rethink the problem</li>
          <li>Parallelization can be more important than model complexity</li>
          <li>Attention mechanisms are incredibly versatile and transferable</li>
        </ul>
      `,
      date: "2025-04-10",
      readTime: "8 min read",
      tags: ["Transformers", "Attention", "NLP", "Deep Learning", "Architecture"],
      type: "literature"
    },
    {
      id: 6,
      title: "Deep Residual Learning for Image Recognition - ResNet",
      excerpt: "Revolutionary paper that solved the vanishing gradient problem with residual connections, enabling training of very deep neural networks...",
      content: `
        <h3>Paper Overview</h3>
        <p><strong>Authors:</strong> Kaiming He et al. (Microsoft Research)</p>
        <p><strong>Published:</strong> CVPR 2016</p>
        <p><strong>arXiv:</strong> <a href="https://arxiv.org/abs/1512.03385" target="_blank">1512.03385</a></p>
        
        <h3>The Problem</h3>
        <p>Before ResNet, training very deep neural networks was problematic due to the vanishing gradient problem. Deeper networks often performed worse than shallower ones, not due to overfitting but due to optimization difficulties.</p>
        
        <h3>The Solution: Residual Learning</h3>
        <p>The key insight was to reformulate the learning problem. Instead of learning the desired underlying mapping H(x), the authors proposed learning the residual F(x) = H(x) - x, and then computing H(x) = F(x) + x.</p>
        
        <p>This simple skip connection allows gradients to flow directly through the identity mapping, solving the vanishing gradient problem.</p>
        
        <h3>Personal Analysis</h3>
        <p>What I find most elegant about ResNet is its simplicity. The residual connection is just an addition operation, yet it enables training networks with hundreds of layers. This demonstrates how small architectural changes can have profound impacts.</p>
        
        <h3>Applications in My Work</h3>
        <p>I've been experimenting with residual connections in some of my projects, particularly in the context of my weight analysis tool. The concept of "shortcuts" in information flow is relevant when analyzing how information propagates through different network architectures.</p>
        
        <h3>Key Insights</h3>
        <ul>
          <li>Sometimes the best solution is the simplest one</li>
          <li>Skip connections enable identity mappings, making optimization easier</li>
          <li>Deeper networks can be better if properly designed</li>
          <li>The residual formulation makes it easier to learn identity functions</li>
        </ul>
        
        <h3>Impact</h3>
        <p>ResNet didn't just win ImageNet 2015 - it fundamentally changed how we think about deep architecture design. Almost every modern architecture incorporates some form of skip connections.</p>
      `,
      date: "2025-01-22",
      readTime: "6 min read",
      tags: ["ResNet", "Computer Vision", "Deep Learning", "Skip Connections", "Architecture"],
      type: "literature"
    },
    {
      id: 7,
      title: "Generative Adversarial Networks - The Original GAN Paper",
      excerpt: "Ian Goodfellow's seminal work introducing the GAN framework that sparked a revolution in generative modeling...",
      content: `
        <h3>Paper Overview</h3>
        <p><strong>Authors:</strong> Ian Goodfellow et al.</p>
        <p><strong>Published:</strong> NIPS 2014</p>
        <p><strong>arXiv:</strong> <a href="https://arxiv.org/abs/1406.2661" target="_blank">1406.2661</a></p>
        
        <h3>The Revolutionary Idea</h3>
        <p>GANs introduced a completely new approach to generative modeling through adversarial training. Two neural networks compete against each other:</p>
        <ul>
          <li><strong>Generator (G):</strong> Learns to create realistic fake data</li>
          <li><strong>Discriminator (D):</strong> Learns to distinguish real from fake data</li>
        </ul>
        
        <h3>The Game Theory Connection</h3>
        <p>The brilliant insight was framing this as a minimax game. The generator tries to minimize the discriminator's ability to detect fakes, while the discriminator tries to maximize its detection accuracy.</p>
        
        <p>Mathematically: <code>min_G max_D E[log D(x)] + E[log(1-D(G(z)))]</code></p>
        
        <h3>Personal Reflections</h3>
        <p>What fascinates me about GANs is how they leverage competition to drive learning. This adversarial principle has inspired me to think about how competition and collaboration might apply to other ML problems.</p>
        
        <p>I've implemented several GAN variants in my projects, including SRGAN for super-resolution. Each time, I'm amazed by how this simple competitive framework can generate such realistic results.</p>
        
        <h3>Challenges & Learnings</h3>
        <p>Training GANs is notoriously difficult - mode collapse, training instability, and convergence issues are common. This taught me valuable lessons about the importance of:</p>
        <ul>
          <li>Careful hyperparameter tuning</li>
          <li>Monitoring multiple metrics during training</li>
          <li>Understanding the theoretical foundations behind the algorithms</li>
        </ul>
        
        <h3>Impact on the Field</h3>
        <p>GANs didn't just introduce a new model - they introduced a new paradigm. The adversarial training concept has been applied far beyond generative modeling, influencing areas like domain adaptation and robustness.</p>
      `,
      date: "2024-12-15",
      readTime: "7 min read",
      tags: ["GANs", "Generative Models", "Adversarial Training", "Deep Learning"],
      type: "literature"
    },
  ],

  certificates: [
    {
      id: 1,
      title: "Advanced Active Learning Techniques",
      description: "I really enjoyed my MNIST active learning project and want to push it further. I want to explore more sophisticated uncertainty sampling methods, test different acquisition functions, and see how far we can push performance with minimal labeled data. Maybe try it on more complex datasets like CIFAR-10 or even some real-world medical imaging.",
      category: "Active Learning",
      status: "Personal Interest",
      estimatedDuration: "6-8 months"
    },
    {
      id: 2,
      title: "Counterfactual Transferability in Adversarial Attacks",
      description: "I want to dive deeper into adversarial examples and specifically explore how counterfactuals can transfer between different models. The goal is to highlight just how dangerous these attacks can be - if we can create adversarial examples that fool multiple models at once, that's a serious security concern that needs more attention.",
      category: "Adversarial ML",
      status: "Research Idea",
      estimatedDuration: "4-6 months"
    },
    {
      id: 3,
      title: "Experimenting with Different RL Algorithms",
      description: "My space navigation project got me hooked on reinforcement learning, but I only tried Deep Q-Learning. I want to experiment with other approaches - PPO, A3C, maybe some more recent algorithms. See which ones work better for different types of problems.",
      category: "Reinforcement Learning",
      status: "Next Project",
      estimatedDuration: "3-4 months"
    },
    {
      id: 4,
      title: "Exploring Advanced GAN Architectures",
      description: "My SRGAN project was just the beginning. I want to test more stable architectures like IDSGAN and other modern variants. Maybe try some conditional GANs or even explore diffusion models more deeply. The goal is to really understand what makes these generative models tick.",
      category: "Generative Models", 
      status: "Future Work",
      estimatedDuration: "5-7 months"
    }
  ]
};

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
  module.exports = portfolioData;
}