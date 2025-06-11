import requests
import json

# Test the agent orchestrator with a text-based query
def test_orchestrator_query():
    url = "http://localhost:8002/api/agent_query"
    
    # Test query for resume collection
    query_data = {
        "query": "What are the key skills mentioned in the resume?",
        "collection": "resume"
    }
    
    print("Testing Agent Orchestrator with resume collection...")
    print(f"Query: {query_data['query']}")
    print(f"Collection: {query_data['collection']}")
    print("-" * 50)
    
    try:
        response = requests.post(url, json=query_data, timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n✅ SUCCESS! Agent Orchestrator Response:")
            print(f"Answer: {result.get('answer', 'No answer provided')}")
            print(f"Confidence: {result.get('confidence', 0):.2f}")
            print(f"Processing Time: {result.get('processing_time', 0):.2f}s")
            print(f"Citations: {result.get('citations', [])}")
            
            # Show evidence details
            evidence = result.get('evidence', {})
            text_evidence = evidence.get('text_evidence', [])
            visual_evidence = evidence.get('visual_evidence', [])
            
            print(f"\nEvidence Summary:")
            print(f"- Text evidence items: {len(text_evidence)}")
            print(f"- Visual evidence items: {len(visual_evidence)}")
            
            # Show agent breakdown
            agent_breakdown = result.get('agent_breakdown', {})
            if agent_breakdown:
                print(f"\nAgent Performance:")
                textual_agent = agent_breakdown.get('textual_agent', {})
                visual_agent = agent_breakdown.get('visual_agent', {})
                
                print(f"- Textual Agent: Confidence {textual_agent.get('confidence', 0):.2f}, "
                      f"Evidence {textual_agent.get('evidence_count', 0)} items, "
                      f"Time {textual_agent.get('processing_time', 0):.2f}s")
                print(f"- Visual Agent: Confidence {visual_agent.get('confidence', 0):.2f}, "
                      f"Evidence {visual_agent.get('evidence_count', 0)} items, "
                      f"Time {visual_agent.get('processing_time', 0):.2f}s")
            
        else:
            print(f"\n❌ ERROR: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Network Error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")

if __name__ == "__main__":
    test_orchestrator_query()
